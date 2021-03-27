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
"""train resnet."""
import os
import argparse
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager

from src.model_skfac import Model_SKFAC as Model
from src.resnet_skfac import resnet50
from src.dataset import create_dataset
from src.crossentropy import CrossEntropy
from src.skfac import SKFAC_GPU as SKFAC
from src.config import config_gpu as config

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--dataset_path', type=str, default='ImageNet2012/train', help='Dataset path')
args_opt = parser.parse_args()

set_seed(1)


def adjust_learning_rate(epoch, batch_idx):
    lr_adj = config.lr_decay ** (epoch + float(batch_idx + 1) / 40036)
    return config.lr_init * lr_adj


def get_model_lr_skfac(total_epoch):
    lr_each_step = []
    for epoch in range(total_epoch):
        for batch_idx in range(40036):
            lr_each_step.append(adjust_learning_rate(epoch, batch_idx))
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[0:]
    return learning_rate


def adjust_damping(epoch, batch_idx):
    epoch += float(batch_idx + 1) / 40036
    return config.damping_init * (config.lr_decay ** (epoch / 10))


def get_model_damping_skfac(total_epoch):
    damping_each_step = []
    for epoch in range(total_epoch):
        for batch_idx in range(40036):
            damping_each_step.append(adjust_damping(epoch, batch_idx))
    damping_each_step = np.array(damping_each_step).astype(np.float32)
    damping_now = damping_each_step[0:]
    return damping_now


if __name__ == '__main__':
    target = 'GPU'
    ckpt_save_dir = config.save_checkpoint_path
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=True, repeat_num=1,
                             batch_size=config.batch_size, target=target)

    # define net
    step_size = dataset.get_dataset_size()
    damping = get_model_damping_skfac(70)
    lr = get_model_lr_skfac(config.lr_end_epoch)
    net = resnet50(class_num=config.class_num, damping=damping, loss_scale=config.loss_scale,
                   frequency=config.frequency, batch_size=config.batch_size)

    # define loss, model
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    opt = SKFAC(filter(lambda x: x.requires_grad, net.get_parameters()), Tensor(lr), config.momentum,
                filter(lambda x: 'matrix_A' in x.name, net.get_parameters()),
                filter(lambda x: 'matrix_G' in x.name, net.get_parameters()),
                filter(lambda x: 'A_inv_max' in x.name, net.get_parameters()),
                filter(lambda x: 'G_inv_max' in x.name, net.get_parameters()),
                config.weight_decay, config.loss_scale)
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    model = Model(net, loss_fn=loss, optimizer=opt, amp_level='O2', loss_scale_manager=loss_scale,
                  keep_batchnorm_fp32=False, metrics={'acc'}, frequency=config.frequency)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]

    # train model
    model.train(config.epoch_size, dataset, callbacks=cb)
