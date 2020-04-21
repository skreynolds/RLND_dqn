#!/usr/bin/env python

'''
NOTE: this script should be run from the RoboND environment
'''

import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# import agent that has been previously created
from dqn_agent import Agent

def main():

	################################################
	# Instantiate the environment and agent
	################################################

	env = gym.make('LunarLander-v2')
	env.seed(0)
	print('State: ', env.observation_space)
	print('State shape: ', env.observation_space.shape)
	print('State high bounds: ', env.observation_space.high)
	print('State low bounds: ', env.observation_space.low)
	print('Action: ', env.action_space)
	print('Number of actions: ', env.action_space.n)

	
	agent = Agent(state_size=8, action_size=4, seed=0)

	# watch the untrained agent
	state = env.reset()	
	img = plt.imshow(env.render(mode='rgb_array'))
	for j in range(200):
		action = agent.act(state)
		img.set_data(env.render(mode='rgb_array'))
		plt.axis('off')
		state, reward, done, _ = env.step(action)
		if done:
			break

	env.close()
	

if __name__ == '__main__':
	main()