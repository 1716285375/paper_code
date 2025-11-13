# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : test_torch.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-10 10:59
@Update Date    : 
@Description    : 
"""
# ------------------------------------------------------------

import torch

def test_torch():
    print('test_torch')
    print('torch version:', torch.__version__)
    print('torch is available:', torch.cuda.is_available())
