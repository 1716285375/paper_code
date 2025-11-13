# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : test_env_magent2.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-13 20:39
@Update Date    : 
@Description    : 
"""
# ------------------------------------------------------------


from environments import make_env



def test_battle_v4_env() -> None:
    env = make_env('magent2:battle_v4')
    print(env)

def test_battle_v4_agent() -> int:
    env = make_env('magent2:battle_v4')
    print(env.agents)

def test_battle_v4_obs_space() -> None:
    pass

def test_battle_v4_action_space() -> None:
    pass

def test_battle_v4_obs_size() -> None:
    pass

def test_battle_v4_action_size() -> None:
    pass