# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import functools
import numpy as np
import os
import torch
import time
import collections
from collections import namedtuple
import gym
import gym_carla
from agent import SAC
import pdb
from env_wrapper import FilterObservationWrapper
import argparse
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

action_repeat = 4
img_stack = 4
env_name = 'carla-v0'
discount = 1.0
number_of_vehicles = 100
number_of_walkers = 0
display_size = 256
max_past_step = 1
dt = 0.1
discrete = False
discrete_acc = [-3.0, 0.0, 3.0]
discrete_steer = [-0.2, 0.0, 0.2]
continuous_accel_range = [-3.0, 3.0]
continuous_steer_range = [-0.3, 0.3]
ego_vehicle_filter = 'vehicle.lincoln*'
port = 2000
town = 'Town03'
task_mode = 'random'
max_time_episode = 500
max_waypt = 12
obs_range = 32
lidar_bin = 0.5
d_behind = 12
out_lane_thres = 2.0
desired_speed = 8
max_ego_spawn_times = 200
display_route = True
pixor_size = 64
pixor = False
obs_channels = None


parser = argparse.ArgumentParser(description='Train a SAC agent for the Carla')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', default=0, action='store_true', help='use visdom')
parser.add_argument(
    '--log-interval', type=int, default=100, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()
env_params = {
    'number_of_vehicles': number_of_vehicles,
    'number_of_walkers': number_of_walkers,
    'display_size': display_size,  # screen size of bird-eye render
    'max_past_step': max_past_step,  # the number of past steps to draw
    'dt': dt,  # time interval between two frames
    'discrete': discrete,  # whether to use discrete control space
    'discrete_acc': discrete_acc,  # discrete value of accelerations
    'discrete_steer': discrete_steer,  # discrete value of steering angles
    'continuous_accel_range': continuous_accel_range,  # continuous acceleration range
    'continuous_steer_range': continuous_steer_range,  # continuous steering angle range
    'ego_vehicle_filter': ego_vehicle_filter,  # filter for defining ego vehicle
    'port': port,  # connection port
    'town': town,  # which town to simulate
    'task_mode': task_mode,  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': max_time_episode,  # maximum timesteps per episode
    'max_waypt': max_waypt,  # maximum number of waypoints
    'obs_range': obs_range,  # observation range (meter)
    'lidar_bin': lidar_bin,  # bin size of lidar sensor (meter)
    'd_behind': d_behind,  # distance behind the ego vehicle (meter)
    'out_lane_thres': out_lane_thres,  # threshold for out of lane
    'desired_speed': desired_speed,  # desired speed (m/s)
    'max_ego_spawn_times': max_ego_spawn_times,  # maximum times to spawn ego vehicle
    'display_route': display_route,  # whether to render the desired route
    'pixor_size': pixor_size,  # size of the pixor labels
    'pixor': pixor,  # whether to output PIXOR observation
}


gym_spec = gym.spec(env_name)
gym_env = gym_spec.make(params=env_params)
gym_env = FilterObservationWrapper(gym_env, ['camera', 'lidar', 'birdeye'], action_repeat, img_stack)

# state = gym_env.reset()

# state_dim = gym_env.observation_space.shape[0]
# action_dim = gym_env.action_space.shape[0]
action_ub = gym_env.action_space.high
action_lb = gym_env.action_space.low
min_Val = torch.tensor(1e-7).float()
Transition = namedtuple('Transition', ['original_s', 's', 'a', 'r', 's_', 'd'])
# 4*64*64
agent = SAC(256, 2, action_ub, action_lb,  min_Val)


training_records = []
total_score = np.array([])
data_for_plot = np.array([])

iters = 0
for i in range(5000):
    if iters >= 2:
        break
    state = gym_env.reset()
    for j in range(1000):
        action_0 = torch.FloatTensor(1, 1).uniform_(action_lb[0], action_ub[0])
        action_1 = torch.FloatTensor(1, 1).uniform_(action_lb[1], action_ub[1])
        action = torch.cat((action_0, action_1), dim=1).T.detach().cpu().numpy()
        state_, reward, done, _ = gym_env.step(action)
        encoded_state = agent.encode_state(torch.FloatTensor(state).to(device).unsqueeze(1))
        encoded_state_ = agent.encode_state(torch.FloatTensor(state_).to(device).unsqueeze(1))
        agent.store(state, encoded_state, action, reward, encoded_state_, done)
        if agent.num_transition % 256 == 0:
            iters += 1
            agent.vae_update()
        state = state_
        if done or j == 999:
            break
print('Pretrain Vae: Done!')

for i_ep in range(10000):
    score = 0
    state = gym_env.reset()
    for t in range(1000):
        # state = torch.FloatTensor(state).cuda()
        # if not state.any():
        #     state = gym_env.reset()
        #     break

        # implement mcts
        # queue = collections.deque()
        # i = 0
        # while i < 3:
        #     action, encoded_state = agent.select_action(state)
        #     state_, reward, done, _ = gym_env.step(action)
        #     queue.append([action, encoded_state, state_, reward, done, None, 0])
        #     i += 1
        # local_optimal_1 = queue[0]
        # local_optimal_2 = queue[0]
        # tmp_score = float('-inf')
        # while queue:
        #     action, encoded_state, state_, reward, done, root, depth = queue.popleft()
        #     if depth > 1:
        #         continue
        #     elif depth == 1:
        #         if reward + root[3] > tmp_score:
        #             local_optimal_1 = root
        #             local_optimal_2 = [action, encoded_state, state_, reward, done]
        #             tmp_score = reward + root[3]
        #     else:
        #         for i in range(3):
        #             tmp0, tmp1 = agent.select_action(state_)
        #             tmp2, tmp3, tmp4, _ = gym_env.step(tmp0)
        #             queue.append([tmp0, tmp1, tmp2, tmp3, tmp4, (action, encoded_state, state_, reward, done), depth+1])
        # score += tmp_score
        # encoded_state_1 = agent.encode_state(torch.FloatTensor(local_optimal_1[2]).to(device).unsqueeze(1))
        # agent.store(state, local_optimal_1[1], local_optimal_1[0], local_optimal_1[3], encoded_state_1,
        #             local_optimal_1[4])
        #
        # encoded_state_2 = agent.encode_state(torch.FloatTensor(local_optimal_2[2]).to(device).unsqueeze(1))
        # agent.store(local_optimal_1[2], local_optimal_2[1], local_optimal_2[0], local_optimal_2[3], encoded_state_2,
        #             local_optimal_2[4])
        action, encoded_state = agent.select_action(state)

        state_, reward, done, _ = gym_env.step(action)
        # state_ = torch.FloatTensor(state_).to(device).unsqueeze(1)
        score += reward
        encoded_state_ = agent.encode_state(torch.FloatTensor(state_).to(device).unsqueeze(1))
        agent.store(state, encoded_state, action, reward, encoded_state_, done)
        if agent.num_transition % 16 == 0:
            agent.update()
        if agent.num_transition % 256 == 0:
            agent.vae_update()
        state = state_
        if done or t == 999:
            break
        # state = local_optimal_2[2]
        # if local_optimal_1[4] or local_optimal_2[4] or t == 999:
        #     break
    total_score = np.append(total_score, score)

    if i_ep % args.log_interval == 0 and i_ep != 0:
        tmp = np.mean(total_score[-args.log_interval:])
        print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, tmp))
        data_for_plot = np.append(data_for_plot, tmp)
        agent.save()

# py_env = gym_wrapper.GymWrapper(
#     gym_env,
#     discount=discount,
#     auto_reset=True,
#   )
np.savetxt('res_16_100_10000_epsilon_greedy', data_for_plot)
plt.plot(data_for_plot)
plt.show()
# eval_py_env = py_env

# if action_repeat > 1:
#   py_env = wrappers.ActionRepeat(py_env, action_repeat)
