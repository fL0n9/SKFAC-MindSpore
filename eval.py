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
import argparse
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.crossentropy import CrossEntropy
from src.config import config_gpu as config
from src.dataset import create_dataset
from src.resnet_skfac import resnet50 as resnet


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str,
                    default='checkpoint_skfac/resnet-{}_{}.ckpt',
                    help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default='ImageNet2012/val', help='Dataset path')
args_opt = parser.parse_args()

set_seed(1)

if __name__ == '__main__':
    target ='GPU'
    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=False, batch_size=config.batch_size,
                             target=target)

    # define net
    net = resnet(class_num=config.class_num)
    net.add_flags_recursive(skfac=False)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    keys = list(param_dict.keys())
    for key in keys:
        if "damping" in key:
            param_dict.pop(key)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
