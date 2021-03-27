# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""SKFAC"""
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator
from mindspore.nn.optim.optimizer import Optimizer


_momentum_opt = C.MultitypeFuncGraph("momentum_opt")

op_add = P.AddN()
apply_decay = C.MultitypeFuncGraph("apply_decay")


@apply_decay.register("Number", "Bool", "Tensor", "Tensor")
def _tensor_apply_decay(weight_decay, if_apply, weight, gradient):
    """Get grad with weight_decay."""
    if if_apply:
        return op_add((weight * weight_decay, gradient))
    return gradient


@_momentum_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt_ext(opt, momentum, learning_rate, gradient, weight, moment):
    """Apply momentum optimizer to the weight parameter using Tensor."""
    success = True
    success = F.depend(success, opt(weight, moment, learning_rate, gradient, momentum))
    return success


class SKFAC_GPU(Optimizer):
    """
    SKFAC
    """

    def __init__(self, params, learning_rate, momentum, matrix_A, matrix_G, A_inv_max, G_inv_max,
                 weight_decay=0.0, loss_scale=1.0, use_nesterov=False, decay_filter=lambda x: x.name not in []):
        super(SKFAC_GPU, self).__init__(learning_rate, params, weight_decay, loss_scale)
        Validator.check_value_type("momentum", momentum, [float], self.cls_name)
        if isinstance(momentum, float) and momentum < 0.0:
            raise ValueError("momentum should be at least 0.0, but got momentum {}".format(momentum))
        self.momentum = Parameter(Tensor(momentum, mstype.float32))
        self.params = self.parameters
        self.use_nesterov = Validator.check_bool(use_nesterov)
        self.moments = self.params.clone(prefix="moments", init='zeros')
        self.hyper_map = C.HyperMap()
        self.opt = P.ApplyMomentum(use_nesterov=self.use_nesterov)
        self.transpose = P.Transpose()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.matmul = P.MatMul()
        self.matrix_A = ParameterTuple(matrix_A)
        self.matrix_G = ParameterTuple(matrix_G)
        self.A_inv_max = ParameterTuple(A_inv_max)
        self.G_inv_max = ParameterTuple(G_inv_max)
        self.assign = P.Assign()
        self.mul = P.Mul()
        self.weight_decay = weight_decay
        self.decay_flags = tuple(decay_filter(x) for x in self.parameters)

    def construct(self, gradients):
        params = self.params
        moments = self.moments
        gradients = self.scale_grad(gradients)
        new_grads = ()
        if self.skfac:
            for i in range(54):
                g = gradients[i * 3]
                g_shape = self.shape(g)
                g = self.reshape(g, (g_shape[0], -1))
                matrix_A = self.matrix_A[i]
                matrix_G = self.matrix_G[i]
                g = self.matmul(self.matmul(matrix_G, g), matrix_A)
                fake_A = self.assign(self.matrix_A[i], matrix_A)
                fake_G = self.assign(self.matrix_G[i], matrix_G)
                g = F.depend(g, fake_A)
                g = F.depend(g, fake_G)
                if i == 53:
                    new_grads = new_grads + (g,)
                else:
                    g = self.reshape(g, g_shape)
                    new_grads = new_grads + (g, gradients[i * 3 + 1], gradients[i * 3 + 2])
        else:
            for i in range(54):
                g = gradients[i * 3]
                g_shape = self.shape(g)
                g = self.reshape(g, (g_shape[0], -1))
                matrix_A = self.matrix_A[i]
                matrix_G = self.matrix_G[i]
                matrix_A = F.depend(matrix_A, g)
                matrix_G = F.depend(matrix_G, g)
                g = self.matmul(self.matmul(matrix_G, g), matrix_A)
                if i == 53:
                    new_grads = new_grads + (g,)
                else:
                    g = self.reshape(g, g_shape)
                    new_grads = new_grads + (g, gradients[i * 3 + 1], gradients[i * 3 + 2])
        gradients = new_grads
        if self.weight_decay > 0:
            gradients = self.hyper_map(F.partial(apply_decay, self.weight_decay), self.decay_flags,
                                       params, gradients)
        lr = self.get_lr()
        success = self.hyper_map(F.partial(_momentum_opt, self.opt, self.momentum, lr), gradients, params, moments)
        return success


