<img src="https://d20vrrgs8k4bvw.cloudfront.net/images/open-graph/udacity.png" width="200" />

# RLND Deep Q-Networks

Implementation of a reinforcement learning agent using a deep q-network. The agent's task is OpenAI's `lunarlander-v2` control problem. The objective is to land a lunar module safely in a designated area.

At each time step the agent receives 8 state inputs:

1. `x-coordinate` - position in the `x` dimension
2. `y-coordinate` - position in the `y` dimension
3. `x-velocity` - velocity in the `x` dimension
4. `y-velocity` - velocity in the `y` dimension
5. 
6. 
7. 
8. 

The agent is allowed to take an action each time step, once it has received the state input. The agent's actions are discrete and include:

1. Do nothing
2. Fire the left facing booster
3. Fire the main thruster
4. Fire the right facing booster

The neural network that was chosen to perform the state to action mapping consisted of 8 input nodes to take the state inputs; four hidden layers of size 16, 32, 16, and 8 nodes; and an output layer that consisted of 4 nodes --- one for each action. 