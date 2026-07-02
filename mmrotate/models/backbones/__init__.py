# Copyright (c) OpenMMLab. All rights reserved.
from .re_resnet import ReResNet

from .pkinet import PKINet
from .lsknet import LSKNet
#from .legnet import MARNet

#__all__ = ['ReResNet', 'PKINet', 'LSKNet', 'MARNet']

from .legnet import LWEGNet


__all__ = ['ReResNet', 'PKINet', 'LSKNet', 'LWEGNet']
