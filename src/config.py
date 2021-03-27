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
"""
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed

# config for resnet50, imagenet2012, GPU
config_gpu = ed({
    "class_num": 1001,
    "batch_size": 32,
    "loss_scale": 1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "epoch_size": 50,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 50,
    "save_checkpoint_path": "./checkpoint_skfac",
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0.00625,
    "lr_decay": 0.87,
    "lr_end_epoch": 50,
    "damping_init": 0.03,
    "frequency": 834,
})
