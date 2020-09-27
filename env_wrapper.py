# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import gym
import numpy as np


class FilterObservationWrapper():
    """Environment wrapper to filter observation channels."""

    def __init__(self, gym_env, input_channels, action_repeat, img_stack):
        super(FilterObservationWrapper, self).__init__(gym_env)
        self.input_channels = input_channels
        self.action_repeat = action_repeat
        self.img_stack = img_stack
        observation_spaces = collections.OrderedDict()
        for channel in self.input_channels:
            observation_spaces[channel] = self._env.observation_space[channel]
        self.observation_space = gym.spaces.Dict(observation_spaces)

    def modify_observation(self, observation):
        observations = collections.OrderedDict()
        for channel in self.input_channels:
            observations[channel] = observation[channel]
        return observations

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
            observation, reward, done, info = self._env.step(action)
            observation = self.modify_observation(observation)
            total_reward += reward
            if done:
                break
        self.stack.pop(0)
        self.stack.append(observation)

        return np.array(self.stack), total_reward, done, info

    def reset(self):
        observation = self._env.reset()
        tmp = self.modify_observation(observation)
        self.stack = [tmp] * self.img_stack
        return np.array(self.stack)
