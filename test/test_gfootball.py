# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : test_gfootball.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-10 10:48
@Update Date    : 
@Description    : 
"""
# ------------------------------------------------------------


import gfootball.env as football_env

def test_env():
     env = football_env.create_environment(
          env_name='11_vs_11_stochastic',
          representation='raw',
          stacked=False,
          logdir='/tmp/football',
          write_goal_dumps=False,
          write_full_episode_dumps=False,
          write_video=False,
          render=False,
          number_of_right_players_agent_controls=1
     )


     obs = env.reset()
     print(obs)
