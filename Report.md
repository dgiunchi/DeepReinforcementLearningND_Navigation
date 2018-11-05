[![chart DQN](https://github.com/dgiunchi/DeepReinforcementLearningND_Navigation/blob/master/chartDQN.png)](#training)

Chart of the loss function for agent using DQN with Vector Observation space size (per agent) 37
The algorithm was DQN, model contains 3 fully connected layers with relu as activation functions.

With DQN and fully connected model the solution came after 412 episodes. That means that the average is over 13 from 412 to 512 episode.

Following the parameters for the model:

DQNetwork(
  (fc1): Linear(in_features=37, out_features=64, bias=True)
  
  (fc2): Linear(in_features=64, out_features=64, bias=True)
  
  (fc3): Linear(in_features=64, out_features=4, bias=True)
  
)

The hyperparameters are the following:

BUFFER_SIZE = int(1e5)  # replay buffer size

BATCH_SIZE = 64         # minibatch size

GAMMA = 0.99            # discount factor

TAU = 1e-3              # for soft update of target parameters

LR = 5e-4               # learning rate 

UPDATE_EVERY = 4        # how often to update the network


The DQN algorithm consist in the epsilon-greedy algorithm, with update of the action-value function after using neural network 
for predicting the better action to take.

The main function that runs the code is in jupyter notebooks while dqnagent.py contains the agent entity with step/act/learn functions.
models.py contains the different network models used in the different examples.

A replay buffer is used for store the experience as stack of states,actions, rewards, nextstates and flags if episode terminates.
This buffer is interrogated when the agent has to learn which action to take. Priority is not yet implemented and a random sample of experiences is used instead.

Models are used to predict the actions to take. DQN use a simple fully connected model, while when images are used as input, convolutional network is used.

Two qnetworks are used in order to have a better estimantion and to avoid instabilities. Local and Target QNetwork are updated in different ways. Target onw a soft update parametrized 
by TAU hyperparameter, simply coping the value from the local qnetwork multiplied by that TAU and its current version of weights.
Local Qnetwork instead is evaluated independently by target. Double DQN makes use of the local qnetwork to estimate the actions additionaly.

While the Pixel DQN changes only the input and the layer of the model, for dueling DQN two branch are created (advantage and value) and then merged for the final estimation.

It is provided also the Solution with the pixel input using DQN with a convolutional neural network and Dueling DQNs.

Future work could be put priority experience.

