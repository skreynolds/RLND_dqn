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

# import agent class
from dqn_agent import Agent

# import the dqn monitoring
from dqn_monitoring import dqn


def main():

	################################################
	# components requred from main_02.py
	################################################

	# spin up environment
	env = gym.make('LunarLander-v2')
	env.seed(0)
	
	# spin up agent (with underlying nn model)
	agent = Agent(state_size=8, action_size=4, seed=0)

	
	################################################
	# Import trained agent and render performance
	################################################

	# load the weights from file
	agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

	for i in range(10):
		state = env.reset()
		img = plt.imshow(env.render(mode='rgb_array'))
		for j in range(400):
			action = agent.act(state)
			img.set_data(env.render(mode='rgb_array'))
			plt.axis('off')
			state, reward, done, _ = env.step(action)
			if done:
				break

	env.close()


if __name__ == '__main__':
	main()