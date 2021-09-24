#!/usr/bin/env python

import argparse
import gym
import safety_gym  # noqa
import numpy as np  # noqa
from safety_gym.envs.engine import Engine
from pprint import pprint

#BOUND = .83
BOUND_S = np.linspace(.83,.9,4)
BOUND_E = BOUND_S[::-1]

def moving(oo):
    moving = np.absolute(oo['accelerometer'][:2]).max() > .4
    return moving

def unsafe(oo):
    n = 4
    if 'hazards_lidar' in oo:
        hl = (oo['hazards_lidar'][:4] > BOUND_S).any() or (oo['hazards_lidar'][-4:] > BOUND_E).any()
    else:
        hl = False
    if 'walls_lidar' in oo:
        wl = (oo['walls_lidar'][:4] > BOUND_S).any() or (oo['walls_lidar'][-4:] > BOUND_E).any()
    else:
        wl = False
    return hl or wl

wall_locs = [
        (-4,4),(-4,3),(-4,2),(-4,1),(-4,0),(-4,-1),(-4,-2),(-4,-3),(-4,-4),
        (-3,4),(-2,4),(-1,4),(0,4),(1,4),(2,4),(3,4),
        ( 4,4),( 4,3),( 4,2),( 4,1),( 4,0),( 4,-1),( 4,-2),( 4,-3),( 4,-4),
        (3,-4),(2,-4),(1,-4),(0,-4),(-1,-4),(-2,-4),(-3,-4),
        ]

def run_random(env_name, render):
    config = {
        'num_steps': 1000,
        'continue_goal': False,
        'observe_remaining': True,
        'observe_goal_comp': False,
        'observe_goal_lidar': True,
        'observe_hazards': True,
        'constrain_hazards': True,
        'goal_size': .4,
        'hazards_cost': 1,
        'hazards_size': .2,
        'hazards_keepout': 0.5,
        'hazards_num': 4,
        'lidar_max_dist': 6,
        'lidar_num_bins': 16,
        'lidar_type': 'pseudo',
        'robot_placements': [(-1, -1, 1, 1)],
        'placements_extents': [-2, -2, 2, 2],
        'robot_base': 'xmls/forward.xml',
        'task': 'sequence',
        'goals_num': 1,
        'reward_distance': 0, # no shaping/only sparse rewards

        'walls_locations': wall_locs,
        'walls_num': len(wall_locs),
#        'walls_num': 0,
        'observe_walls': True,
        '_seed': 0,
    }
    env = Engine(config)
    obs, original_obs = env.reset()
    done = False
    ep_ret = 0
    ep_cost = 0
    step = 0
    state = 0
    while True:
        if done:
            print('Episode Return: %.3f \t Episode Cost: %.3f'%(ep_ret, ep_cost))
            ep_ret, ep_cost,step = 0, 0, 0 
            obs, original_obs = env.reset()
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        if unsafe(original_obs):
            if state == 0 and moving(original_obs):
                state = 1
                act[1] = 0.0
            if state == 1 and not moving(original_obs):
                state = 2
                act[1] = abs(act[1])
            if state == 2:
                act[1] = abs(act[1])
            act[0] = env.action_space.low[0]
        else:
            state = 0
            act[0] = env.action_space.high[0]
        obs, original_obs, reward, done, info = env.step(act)
        ep_ret += reward
        ep_cost += info.get('cost', 0)
        if render:
            env.render()
        step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Safexp-PointGoal1-v0')
    parser.add_argument('--render', action='store_true', default=False)
    args = parser.parse_args()
    run_random(args.env, args.render)
