# ================================================================================
# Copyright 2018 Alibaba Inc. All Rights Reserved.
# ================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from typing import List
from silkflow_framework.sdk.base_pipeline import BasePipeline
from silkflow_framework.sdk.expt_manager import ExptManager


class PipelineTemplate(BasePipeline):
    """A pipeline template to run train"""

    def __init__(self, expt_manager: ExptManager, prefix: str, dep_ops: List = [], configs: dict = {},
                 params: dict = {}):
        if not prefix:
            prefix = self.__class__.__name__
        default_params = {}
        # update experiment directory.
        silkflow_detail_dir = '%s/silkflow_detail' % (os.getcwd() if expt_manager is None else expt_manager.expt_dir)
        if not expt_manager or not hasattr(expt_manager, '_update_runtime_status'):
            expt_manager = ExptManager(expt_dir=silkflow_detail_dir)
        else:
            expt_manager._update_runtime_status(expt_dir=silkflow_detail_dir)
        # init base pipeline.
        super()._init(expt_manager, prefix, dep_ops=dep_ops, configs=configs, params=params,
                      default_params=default_params)

    def _define(self):
        action_op = super()._add_op(name="GLAT-main-20220104-16_13_30",
                                    image="reg.docker.alibaba-inc.com/silkflow/pytorch:1.6.0-cuda10.1-cudnn7-runtime-fairseq",
                                    command="cd /mnt/nas/users/zhanjiaao.zja/workspace/gitlab.alibaba-inc.com/zhanjiaao.zja/my-code/GLAT-main;bash train_vanilla_enro_left.sh",
                                    gpus=4,
                                    cpu=5,
                                    memory=32,
                                    requirements="pip install apex -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com; pip install -U numpy matplotlib pandas omegaconf==2.0.5 hydra-core==1.0.4 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com ",
                                    node_selector={})
        self.last_ops += action_op