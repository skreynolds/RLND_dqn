# import required libraries 
import torch
import numpy as np
from collections import deque

# function initiates agent learning in environment
def dqn(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
	""" Deep Q-Learning

	Params
	======
	
	- env (gym class):
	- agent (Agent class):
	- n_episodes (int): maximum number of training episodes
	- mat_t (int): maximum number of timesteps per episode
	- eps_start (float): starting value of epsilon, for epsilon-greedy action selection
	- eps_end (float): minimum value of epsilon
	- eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
	
	"""

	scores = [] 							# list containing scores from each episode
	scores_window = deque(maxlen=100)		# last 100 scores
	eps = eps_start							# initialise epsilon

	for i_episode in range(1, n_episodes+1):
		
		# initalise episode
		state = env.reset()
		score = 0

		# run episode
		for t in range(max_t):
			action = agent.act(state, eps)
			next_state, reward, done, _ = env.step(action)
			agent.step(state, action, reward, next_state, done)
			state = next_state
			score += reward
			if done:
				break
		
		# save scores and change eps
		scores_window.append(score)			# save most recent score
		scores.append(score)				# save most recent score
		eps = max(eps_end, eps_decay*eps)	# decrease epsilon

		# print periodic information to terminal
		print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
		if i_episode % 100 == 0:
			print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

		# conclude simulation when problem is solved
		if np.mean(scores_window)>=200.0:
			print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
			torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
			break
		
	return scores