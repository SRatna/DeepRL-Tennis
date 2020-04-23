### Learning Algorithm

In this project, I have used MADDPG algorithm to train the agents.
- As we have two agents in this tennis game, each agent acts as actor-critic on its own.
- That means each agent will have total of 4 networks: actor and critic networks, both with different target networks too.
- Thus, if we sum up there will be 8 networks in total. And these networks will collaborate and compete at the same time for maximum gain of rewards.
- During the training phase, traget network of actor is used to obtain the targeted Q-values for critic while local critic network is used to get expected Q-values.
- Then, critic loss is calculated using MSE loss fuction.
- After that, backpropagation and optimization steps are performed for the critic network.
- As for the actor network, we used the critic network, which is essentially an action-value network, to find the loss for actor network.
- Once loss is calculated, backpropagation(via gradient ascent) and optimization steps are performed for the actor network as well.
- And finally, traget networks'(of both actor and critic) model parameters are changed using soft update method.
- Basically, this training phase is similar to DDPG algorithm only difference is that both independent agents will share common replay memory where experiences from both agents will be added, and during training phase those experiences will be sampled. This makes sure that both agents learn from the exerience of each others, thus the collaboration.

### Hyperparameters

Following hyperparameters are used in the project:
* BUFFER_SIZE = int(1e6) : replay buffer size
* BATCH_SIZE = 256 : minibatch size
* GAMMA = 0.95 : discount factor
* TAU = 1e-3 : for soft update of target parameters
* LR_ACTOR = 1e-4 : learning rate of the actor 
* LR_CRITIC = 1e-3 : learning rate of the critic
* WEIGHT_DECAY = 0 : L2 weight decay
* EPSILON = 1.0 : epsilon for the noise process added to the actions
* EPSILON_DECAY = 1e-6 : decay for epsilon above

### Network Architecture

3 Fully connected linear layers are used in both local and target networks of both actor and critic. Relu activation fuctions are used in the outputs of input and hidden layers. Batch normalization is also performed to the output of first input layer before it is passed through relu fuction. Final output of actor networks is also passed through tahn activation function as we need the output to be in range of -1 and 1.
As for actor networks, first input layer is of size 24 x 64. Likewise hidden layer is of size 64 x 64 and finally output layer is of size 64 x 2. Here 24 is the number of states and 2 is number of actions. 
As for critic networks, first input layer is of size 24 x 64. Likewise hidden layer is of size 66 x 64 and finally output layer is of size 64 x 1. Here 24 is the number of states and 2 is number of actions. We have 66 as size of hidden layer as the output of first input layer is concatinated with the action values (which is of size 2).

### Plot of Rewards

#### Training Phase

I trained the agents for 2000 episodes and obtained an average score of 1.0248.

![Scores during training](https://raw.githubusercontent.com/SRatna/DeepRL-Tennis/master/plots/train.png)

#### Training Phase

The average score over 100 epsoides is 2.37 during testing phase.

![Scores during testing](https://raw.githubusercontent.com/SRatna/DeepRL-Tennis/master/plots/test.png)

### Future Work

Some of advanced algorithms can be used to obtain better performance. A few of them are:

* PPO: proximal Policy Optimization algorithm
* A2C: Advatage Actor Critic algorithm
* GAE: Generalized Advantage Estimation algorithm
* Prioritized Experience Replay

Moreover, hyperparemeters like buffer size, learning rate and discouting rate can be changed and we can hope more improvement through it. Also, the depth of neural networks can be increase although it might increase the training time. We can also try different optimizers and see its effect in the overall performance of the agent. 

Ofcourse trying out more complex environment like Play Soccer would be a part of future work too.