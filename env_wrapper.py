from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import gym
import numpy as np


class FilterObservationWrapper():
    """Environment wrapper to filter observation channels."""

    def __init__(self, gym_env, input_channels, action_repeat, img_stack):
        # self.input_channels = input_channels
        self.action_repeat = action_repeat
        self.img_stack = img_stack
        self.gym_env = gym_env
        # observation_spaces = collections.OrderedDict()
        # for channel in self.input_channels:
        #     observation_spaces[channel] = self.gym_env.observation_space[channel]
        # self.observation_space = gym.spaces.Dict(observation_spaces)
        self.action_space = self.gym_env.action_space

    # def modify_observation(self, observation):
    #     observations = collections.OrderedDict()
    #     for channel in self.input_channels:
    #         observations[channel] = observation[channel]
    #     return observations

    def rgb2gray(self, rgb, norm=True):
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            gray = gray/128.-1
        return gray

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
            observation, reward, done, info = self.gym_env.step(action)
            # observation = self.modify_observation(observation)
            total_reward += reward
            if done:
                break
        gray_camera = self.rgb2gray(observation['camera'])
        gray_lidar = self.rgb2gray(observation['lidar'])
        self.stack.pop(0)
        self.stack.append(gray_camera+gray_lidar)

        return np.array(self.stack), total_reward, done, info

    def reset(self):
        observation = self.gym_env.reset()
        # print(observation['camera'].shape)
        gray_camera = self.rgb2gray(observation['camera'])
        gray_lidar = self.rgb2gray(observation['lidar'])
        # tmp = self.modify_observation(observation)
        self.stack = [gray_camera+gray_lidar] * self.img_stack
        return np.array(self.stack)
