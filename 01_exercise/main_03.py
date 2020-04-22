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
	# Train the Agent with DQN
	################################################

	# train the agent
	scores = dqn(env, agent)

	# plot the scores that the agent received while training
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(np.arange(len(scores)), scores)
	plt.xlabel('Episode #')
	plt.ylabel('Score')
	plt.show()


if __name__ == '__main__':
	main()