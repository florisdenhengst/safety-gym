#!/usr/bin/env python

import argparse
import gym
import safety_gym  # noqa
import numpy as np  # noqa
from safety_gym.envs.engine import Engine

BOUND = .85

def unsafe(oo):
    return (oo['hazards_lidar'][:3] > BOUND).any() or (oo['hazards_lidar'][-3:] > BOUND).any()

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
        'hazards_num': 5,
        'lidar_max_dist': 6,
        'lidar_num_bins': 16,
        'lidar_type': 'pseudo',
        'robot_placements': [(-1, -1, 1, 1)],
        'placements_extents': [-2, -2, 2, 2],
        'robot_base': 'xmls/forward.xml',
        'task': 'sequence',
        'goals_num': 1,
        'reward_distance': 0, # no shaping/only sparse rewards
    }
    env = Engine(config)
    obs, original_obs = env.reset()
    done = False
    ep_ret = 0
    ep_cost = 0
    step = 0
    while True:
        if done:
            print('Episode Return: %.3f \t Episode Cost: %.3f'%(ep_ret, ep_cost))
            ep_ret, ep_cost,step = 0, 0, 0 
            obs, original_obs = env.reset()
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        if unsafe(original_obs):
            act[0] = env.action_space.low[0]
            act[1] = abs(act[1])
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
