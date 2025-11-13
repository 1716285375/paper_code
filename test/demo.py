# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : demo.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-12 16:19
@Update Date    : 
@Description    : 
"""
# ------------------------------------------------------------


from magent2.environments import battle_v4

env = battle_v4.parallel_env(map_size=12)
obs = env.reset()
print(env.agents)