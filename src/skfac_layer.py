"""skfac_layer"""
import numpy as np
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator, twice
from mindspore._extends import cell_attr_register
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.nn.layer.activation import get_activation
from mindspore.ops import operations as P
from mindspore import log as logger



class _Conv(Cell):
    r"""Applies a N-D convolution over an input signal composed of several input
       planes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 pad_mode,
                 padding,
                 dilation,
                 group,
                 data_format,
                 has_bias,
                 weight_init,
                 bias_init,
                 ):
        super(_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.padding = padding
        self.dilation = dilation
        self.group = group
        self.data_format = data_format
        self.has_bias = has_bias
        if not (isinstance(in_channels, int) and in_channels > 0):
            raise ValueError('Attr \'in_channels\' of \'Conv2D\' Op passed '
                             + str(in_channels) + ', should be a int and greater than 0.')
        if (not isinstance(kernel_size, tuple)) or len(kernel_size) != 2 or \
                (not isinstance(kernel_size[0], int)) or (not isinstance(kernel_size[1], int)) or \
                kernel_size[0] < 1 or kernel_size[1] < 1:
            raise ValueError('Attr \'kernel_size\' of \'Conv2D\' Op passed '
                             + str(self.kernel_size) + ', should be a int or tuple and equal to or greater than 1.')
        if in_channels % group != 0:
            raise ValueError('Attr \'in_channels\' of \'Conv2D\' Op must be divisible by '
                             'attr \'group\' of \'Conv2D\' Op.')
        if out_channels % group != 0:
            raise ValueError('Attr \'out_channels\' of \'Conv2D\' Op must be divisible by '
                             'attr \'group\' of \'Conv2D\' Op.')

        self.weight = Parameter(initializer(
            weight_init, [out_channels, in_channels // group, *kernel_size]))

        if Validator.check_bool(has_bias):
            self.bias = Parameter(initializer(bias_init, [out_channels]))
        else:
            if bias_init != 'zeros':
                logger.warning("Value of 'has_bias' is False, value of 'bias_init' will be ignored.")
            self.bias = None

    def construct(self, *inputs):
        raise NotImplementedError


class Conv2d_SKFAC_GPU(_Conv):
    """Conv2d_SKFAC"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 data_format='NCHW',
                 has_bias=False,
                 weight_init='normal',
                 damping=0.03,
                 loss_scale=1,
                 frequency=278,
                 batch_size=32,
                 bias_init='zeros'):
        self.skfac = True
        self.hw = kernel_size * kernel_size
        kernel_size = twice(kernel_size)
        super(Conv2d_SKFAC_GPU, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            data_format,
            has_bias,
            weight_init,
            bias_init,
        )
        self.conv2d = P.Conv2D(out_channel=self.out_channels,
                               kernel_size=self.kernel_size,
                               mode=1,
                               pad_mode=self.pad_mode,
                               pad=self.padding,
                               stride=self.stride,
                               dilation=self.dilation,
                               group=self.group
                               )

        self.matrix_A_dim = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.matrix_G_dim = self.out_channels
        split_dim = 128
        self.matrix_A_inv = Parameter(np.zeros((self.matrix_A_dim, self.matrix_A_dim)).astype(np.float32),
                                      requires_grad=False)
        self.matrix_G_inv = Parameter(np.zeros((self.matrix_G_dim, self.matrix_G_dim)).astype(np.float32),
                                      requires_grad=False)

        self.cov_step = Parameter(initializer(0, [1], mstype.int32), requires_grad=False)
        self.img2col = P.Im2Col(kernel_size=kernel_size, stride=stride,
                                pad_mode="same")
        self.matmul = P.MatMul(transpose_a=True)
        self.matmul_ = P.MatMul()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.mul = P.Mul()
        self.getG = P.InsertGradientOf(self.save_gradient)
        self.loss_scale = Tensor(1 / loss_scale, mstype.float16)
        self.batch_size = Tensor(batch_size, mstype.float16)
        self.transpose = P.Transpose()
        self.cast = P.Cast()
        self.gather = P.Gather()
        self.freq = Tensor(frequency, mstype.int32)
        self.axis = 0
        self.sqrt = P.Sqrt()
        self.reduce_mean = P.ReduceMean(keep_dims=False)
        self.damping = Parameter(Tensor(damping), requires_grad=False)
        self.dampingA = Tensor(np.identity(batch_size), mstype.float32)
        self.dampingG = Tensor(np.identity(batch_size), mstype.float32)
        self.I_G = Tensor(np.identity(out_channels), mstype.float32)
        self.I_A = Tensor(np.identity(self.matrix_A_dim), mstype.float32)
        self.cholesky = P.CholeskyTrsm(split_dim=split_dim)
        self.vector_matmul = P.BatchMatMul(transpose_a=True)
        self.batch_coefficient = Tensor((1 / 32) ** 0.5, mstype.float32)

    def save_gradient(self, dout):
        """save_gradient"""
        '''loss_scale=1'''
        """SKFAC compute_cov"""
        out = dout
        dout = self.mul(dout, self.batch_size)
        dout_shape = self.shape(dout)
        dout = self.transpose(dout, (1, 0, 2, 3))  # [out_channels, batch_size, sp, sp]
        dout = self.reshape(dout, (dout_shape[1], dout_shape[0], -1))  # [out_channels, batch_size, sp*sp]
        dout = self.reduce_mean(dout, 2)  # [out_channels, batch_size]
        dout_shape = self.shape(dout)
        dout = self.cast(dout, mstype.float32)
        dout = self.mul(dout, self.batch_coefficient)
        """SKFAC compute_cov finished"""
        damping_step = self.gather(self.damping, self.cov_step, 0)
        damping_step = self.cast(damping_step, mstype.float32)
        self.cov_step = self.cov_step + self.freq
        damping = self.sqrt(damping_step)
        '''SKFAC inv_cov'''
        dout_t = self.transpose(dout, (1, 0))  # [batch_size, out_channels]
        dout_cov = self.matmul_(dout_t, dout)
        damping_G = self.mul(self.dampingG, damping)
        dout_cov = dout_cov + damping_G
        dout_cov_inv = self.cholesky(dout_cov)  # [1, batch_size, batch_size]
        dout_cov_inv = self.reshape(dout_cov_inv, (dout_shape[1], dout_shape[1]))
        dout_cov_inv = self.matmul(dout_cov_inv, dout_cov_inv)
        dout_cov_inv = self.mul(self.I_G - self.matmul_(dout, self.matmul_(dout_cov_inv, dout_t)), 1 / damping)
        self.matrix_G_inv = dout_cov_inv  # [out_channels, out_channels]
        return out

    def construct(self, x):
        if self.skfac:
            '''SKFAC compute_conv'''
            matrix_A = self.img2col(x)  # (in_channels, kernel_size, kernel_size, batch_size, sp, sp)
            matrix_A_shape = self.shape(matrix_A)
            matrix_A = self.reshape(matrix_A, (matrix_A_shape[0] * matrix_A_shape[1] * matrix_A_shape[2],
                                               matrix_A_shape[3], -1))
            matrix_A = self.reduce_mean(matrix_A, 2)  # (in_channels*kernel_size**2, batch size)
            matrix_A_shape = self.shape(matrix_A)
            matrix_A = self.cast(matrix_A, mstype.float32)
            matrix_A = self.mul(matrix_A, self.batch_coefficient)
            '''SKFAC compute_conv finished'''
            damping_step = self.gather(self.damping, self.cov_step, 0)
            damping_step = self.cast(damping_step, mstype.float32)
            self.cov_step = self.cov_step + self.freq
            damping = self.sqrt(damping_step)
            '''SKFAC inv_cov'''
            matrix_A_t = self.transpose(matrix_A, (1, 0))
            matrix_A_cov = self.matmul_(matrix_A_t, matrix_A)
            damping_A = self.mul(self.dampingA, damping)
            matrix_A_cov = matrix_A_cov + damping_A
            matrix_A_cov_inv = self.cholesky(matrix_A_cov)
            matrix_A_cov_inv = self.reshape(matrix_A_cov_inv, (matrix_A_shape[1], matrix_A_shape[1]))
            matrix_A_cov_inv = self.matmul(matrix_A_cov_inv, matrix_A_cov_inv)
            matrix_A_cov_inv = self.mul(self.I_A - self.matmul_(matrix_A, self.matmul_(matrix_A_cov_inv, matrix_A_t)),
                                        1 / damping)
            self.matrix_A_inv = matrix_A_cov_inv
            out = self.conv2d(x, self.weight)
            out = self.getG(out)
        else:
            out = self.conv2d(x, self.weight)

        return out

    def extra_repr(self):
        """extra_repr"""
        s = 'input_channels={}, output_channels={}, kernel_size={},' \
            'stride={},  pad_mode={}, padding={}, dilation={}, ' \
            'group={}, data_format={}, has_bias={},' \
            'weight_init={}, bias_init={}'.format(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.pad_mode,
            self.padding,
            self.dilation,
            self.group,
            self.data_format,
            self.has_bias,
            self.weight,
            self.bias)

        if self.has_bias:
            s += ', bias={}'.format(self.bias)
        return s


class Dense_SKFAC_GPU(Cell):
    """Dense_SKFAC"""

    @cell_attr_register(attrs=['has_bias', 'activation'])
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 damping=0.03,
                 loss_scale=1,
                 frequency=278,
                 batch_size=32,
                 has_bias=True,
                 activation=None):
        super(Dense_SKFAC_GPU, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.has_bias = Validator.check_bool(has_bias)
        self.skfac = True
        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError("weight_init shape error")

        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels]))

        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError("bias_init shape error")

            self.bias = Parameter(initializer(bias_init, [out_channels]))

        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()
        split_dim = 128
        self.activation = get_activation(activation)
        self.activation_flag = self.activation is not None
        self.matrix_A_inv = Parameter(Tensor(np.zeros((in_channels, in_channels)).astype(np.float32)),
                                      requires_grad=False)
        self.matrix_G_inv = Parameter(Tensor(np.zeros((out_channels, out_channels)).astype(np.float32)),
                                      requires_grad=False)
        self.cov_step = Parameter(initializer(0, [1], mstype.int32), requires_grad=False)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.mul = P.Mul()
        self.matmul = P.MatMul(transpose_a=True)
        self.matmul_B = P.MatMul(transpose_b=True)
        self.matmul_ = P.MatMul()
        self.loss_scale = Tensor(1 / loss_scale, mstype.float16)
        self.batch_size = Tensor(batch_size, mstype.float16)
        self.getG = P.InsertGradientOf(self.save_gradient)
        self.damping = Parameter(Tensor(damping), requires_grad=False)
        self.dampingA = Tensor(np.identity(batch_size), mstype.float32)
        self.dampingG = Tensor(np.identity(batch_size), mstype.float32)
        self.I_G = Tensor(np.identity(out_channels), mstype.float32)
        self.I_A = Tensor(np.identity(in_channels), mstype.float32)
        self.cast = P.Cast()
        self.gather = P.Gather()
        self.freq = Tensor(frequency, mstype.int32)
        self.axis = 0
        self.add = P.Add()
        self.sqrt = P.Sqrt()
        self.cholesky = P.CholeskyTrsm(split_dim=split_dim)
        self.vector_matmul = P.BatchMatMul(transpose_a=True)
        self.batch_coefficient = Tensor((1 / 32) ** 0.5, mstype.float32)

    def save_gradient(self, dout):
        """save_gradient"""
        '''SKFAC compute_cov'''
        out = dout  # [batch size, out_channel]
        dout = self.mul(dout, self.batch_size)
        dout_shape = self.shape(dout)
        dout = self.cast(dout, mstype.float32)
        dout = self.transpose(dout, (1, 0))  # [channel,batch size]
        dout = self.mul(dout, self.batch_coefficient)
        '''SKFAC compute_cov finished'''
        damping_step = self.gather(self.damping, self.cov_step, 0)
        damping_step = self.cast(damping_step, mstype.float32)
        self.cov_step = self.cov_step + self.freq
        damping = self.sqrt(damping_step)
        '''SKFAC inv_cov'''
        dout_t = self.transpose(dout, (1, 0))  # [batch_size, channel]
        dout_cov = self.matmul_(dout_t, dout)
        damping_G = self.mul(self.dampingG, damping)
        dout_cov = dout_cov + damping_G
        dout_cov_inv = self.cholesky(dout_cov)  # [1, batch_size, batch_size]
        dout_cov_inv = self.reshape(dout_cov_inv, (dout_shape[0], dout_shape[0]))
        dout_cov_inv = self.matmul(dout_cov_inv, dout_cov_inv)
        dout_cov_inv = self.mul(self.I_G - self.matmul_(dout, self.matmul_(dout_cov_inv, dout_t)), 1 / damping)
        self.matrix_G_inv = dout_cov_inv
        return out

    def construct(self, x):
        """construct"""
        if self.skfac:
            '''SKFAC compute_cov'''
            inputs = x
            inputs = self.cast(inputs, mstype.float32)
            inputs_shape = self.shape(inputs)
            normalizer = inputs_shape[0]
            inputs = self.transpose(inputs, (1, 0))  # [channel, batch size]
            matrix_A = self.mul(inputs, self.batch_coefficient)
            '''SKFAC compute_cov end'''
            damping_step = self.gather(self.damping, self.cov_step, 0)
            damping_step = self.cast(damping_step, mstype.float32)
            self.cov_step = self.cov_step + self.freq
            damping = self.sqrt(damping_step)
            '''SKFAC inv_cov'''
            matrix_A_t = self.transpose(matrix_A, (1, 0))
            matrix_A_cov = self.matmul_(matrix_A_t, matrix_A)
            damping_A = self.mul(self.dampingA, damping)
            matrix_A_cov = matrix_A_cov + damping_A
            matrix_A_cov_inv = self.cholesky(matrix_A_cov)
            matrix_A_cov_inv = self.reshape(matrix_A_cov_inv, (inputs_shape[0], inputs_shape[0]))
            matrix_A_cov_inv = self.matmul(matrix_A_cov_inv, matrix_A_cov_inv)
            matrix_A_cov_inv = self.mul(self.I_A - self.matmul_(matrix_A, self.matmul_(matrix_A_cov_inv, matrix_A_t)),
                                        1 / damping)
            self.matrix_A_inv = matrix_A_cov_inv
            output = self.matmul_B(x, self.weight)
            output = self.getG(output)
        else:
            output = self.matmul_B(x, self.weight)

        if self.has_bias:
            output = self.bias_add(output, self.bias)
        if self.activation_flag:
            return self.activation(output)
        return output

    def extend_repr(self):
        """extend_repr"""
        s = 'in_channels={}, out_channels={}'.format(self.in_channels, self.out_channels)
        if self.has_bias:
            s += ', has_bias={}'.format(self.has_bias)
        if self.activation_flag:
            s += ', activation={}'.format(self.activation)
        return s


