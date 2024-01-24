import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from itertools import count
import matplotlib.pyplot as plt
import random
import copy
import gym
import time
from tqdm import tqdm

print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Gym version: {gym.__version__}")

from hopperenv import *     # Our Rocket Hopper Environment
from helperfunc import *     # helper functions for plotting etc

## INITIALIZATION OF HYPERPARAMETERS	
BUFFER_SIZE=1_000_000 # Buffer size of 1 million entrie
BATCH_SIZE=256 #64   # Sampling from memory - This can be 128 for more complex tasks such as Hopper
UPDATE_ITERATION=10 # Number of iterations in the replay buffer

tau=0.01/2    #0.01       # Target Network HyperParameters (soft updating)
gamma=0.99      # ?
directory = './'

# Neural Network architecture:
H1=16     #32  # Neuron of 1st Layers #400 #20 # 64
H2=27     #H1*2  # Neurons of 2nd layers #300 #64 # 128

# TD3 Agent Parameters:
learning_rate_actor = 0.00024188
learning_rate_critic = 0.0001
policy_freq = 2                 # Frequency of delayed policy updates

# Agent Training Parameters:
MAX_EPISODE=50   # Number of episodes 200
ep_r = 0          # Initial episode reward: normally 0 or -infinity

## EXPLORATION NOISE ======================================================##

class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def generate(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    

## REPLAY BUFFER ==========================================================##
class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=BUFFER_SIZE):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, BATCH_SIZE):
        """Sample a batch of experiences.
        Parameters
        ----------
        BATCH_SIZE: int
            How many transitions to sample.
        Returns
        -------
        state: np.array
            batch of state or observations
        action: np.array
            batch of actions executed given a state
        reward: np.array
            rewards received as results of executing action
        next_state: np.array
            next state next state or observations seen after executing action
        done: np.array
            done[i] = 1 if executing ation[i] resulted in
            the end of an episode and 0 otherwise.
        """
        ind = np.random.randint(0, len(self.storage), size=BATCH_SIZE)
        state, next_state, action, reward, done = [], [], [], [], []

        for i in ind:
            st, n_st, act, rew, dn = self.storage[i]
            state.append(np.array(st, copy=False))
            next_state.append(np.array(n_st, copy=False))
            action.append(np.array(act, copy=False))
            reward.append(np.array(rew, copy=False))
            done.append(np.array(dn, copy=False))

        return np.array(state), np.array(next_state), np.array(action), np.array(reward).reshape(-1, 1), np.array(done).reshape(-1, 1)

## NETWORK ARCHITECTURE ===================================================##
class Actor(nn.Module):
    """
    The Actor model takes in a state observation as input and 
    outputs an action, which is a continuous value.
    
    It consists of four fully coonected linear layers with ReLU activation functions and 
    a final output layer selects one single optimized action for the state
    """
    def __init__(self, n_states, action_dim, H1):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, H1), 
            nn.ReLU(), 
            nn.Linear(H1, H1), 
            nn.ReLU(), 
            nn.Linear(H1, H1), 
            nn.ReLU(), 
            nn.Linear(H1, action_dim),
            nn.Tanh() 
        )
        
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    """
    The Critic model takes in both a state observation and an action as input and 
    outputs a Q-value, which estimates the expected total reward for the current state-action pair. 
    
    It consists of four linear layers with ReLU activation functions, 
    State and action inputs are concatenated before being fed into the first linear layer. 
    
    The output layer has a single output, representing the Q-value
    """
    def __init__(self, n_states, action_dim, H2):
        super(Critic, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(n_states + action_dim, H2), 
            nn.ReLU(), 
            nn.Linear(H2, H2), 
            nn.ReLU(), 
            nn.Linear(H2, H2), 
            nn.ReLU(), 
            nn.Linear(H2, action_dim),
            nn.Tanh()
        )

        self.q2 = nn.Sequential(
            nn.Linear(n_states + action_dim, H2), 
            nn.ReLU(), 
            nn.Linear(H2, H2), 
            nn.ReLU(), 
            nn.Linear(H2, H2), 
            nn.ReLU(), 
            nn.Linear(H2, action_dim),
            nn.Tanh()
        )

    def forward(self, state, action):
        return self.q1(torch.cat((state, action), 1)), self.q2(torch.cat((state, action), 1))
    
    def Q1(self, state, action):
        return self.q1(torch.cat((state, action), 1))

## GPU SETUP ===============================================================##
# Set GPU for faster training (if available)
cuda = torch.cuda.is_available() #check for CUDA
device   = torch.device("cuda" if cuda else "cpu")
print("Job will run on {}".format(device))


## TD3 AGENT ===============================================================##
class TD3(object):
    def __init__(self, state_dim, action_dim):
        """
        Initializes the DDPG agent. 
        Takes three arguments:
               state_dim which is the dimensionality of the state space, 
               action_dim which is the dimensionality of the action space, and 
               max_action which is the maximum value an action can take. 
        
        Creates a replay buffer, an actor-critic  networks and their corresponding target networks. 
        It also initializes the optimizer for both actor and critic networks alog with 
        counters to track the number of training iterations.
        """
        self.replay_buffer = Replay_buffer()
        #print(f"learning rates: actor {learning_rate_actor} | critic {learning_rate_critic}")
        
        self.actor = Actor(state_dim, action_dim, H1).to(device)
        self.actor_target = Actor(state_dim, action_dim,  H1).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)

        self.critic = Critic(state_dim, action_dim,  H2).to(device)
        self.critic_target = Critic(state_dim, action_dim,  H2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate_critic)
        # learning rate

        self.policy_freq = policy_freq

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0


    def select_action(self, state):
        """
        takes the current state as input and returns an action to take in that state. 
        It uses the actor network to map the state to an action.
        """
        #print(f"Doing an Action: state {state} reshaped {state.reshape(1, -1)}")
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # print(state) # tensor([[0., 0.]])
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        """
        updates the actor and critic networks using a batch of samples from the replay buffer. 
        For each sample in the batch, it computes the target Q value using the target critic network and the target actor network. 
        It then computes the current Q value using the critic network and the action taken by the actor network. 
        
        It computes the critic loss as the mean squared error between the target Q value and the current Q value, and 
        updates the critic network using gradient descent. 
        
        It then computes the actor loss as the negative mean Q value using the critic network and the actor network, and 
        updates the actor network using gradient ascent. 
        
        Finally, it updates the target networks using soft updates, where a small fraction of the actor and critic network weights 
        are transferred to their target counterparts. 
        This process is repeated for a fixed number of iterations.
        """

        for it in range(UPDATE_ITERATION):
            # For each Sample in replay buffer batch
            state, next_state, action, reward, done = self.replay_buffer.sample(BATCH_SIZE)
            state = torch.FloatTensor(state).to(device)
            #print(state)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(1-done).to(device)
            reward = torch.FloatTensor(reward).to(device)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state,self.actor_target(next_state))
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * gamma * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss as the negative mean Q value using the critic network and the actor network
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            if self.num_actor_update_iteration % self.policy_freq == 0:
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                
                """
                Update the frozen target models using 
                soft updates, where 
                tau,a small fraction of the actor and critic network weights are transferred to their target counterparts. 
                """
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
           
            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1
            
    def save(self,filename=None):
        """
        Saves the state dictionaries of the actor and critic networks to files
        """
        if filename is None:
            torch.save(self.actor.state_dict(), directory + 'actor.pth')
            torch.save(self.critic.state_dict(), directory + 'critic.pth')
        else:
            torch.save(self.actor.state_dict(), directory + f'actor_{filename}.pth')
            torch.save(self.critic.state_dict(), directory + f'critic_{filename}.pth')
        

    def load(self,filename=None):
        """
        Loads the state dictionaries of the actor and critic networks to files
        """
        if filename is None:
            self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
            self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        else:
            self.actor.load_state_dict(torch.load(directory + f'actor_{filename}.pth'))
            self.critic.load_state_dict(torch.load(directory + f'critic_{filename}.pth'))
       
## INITIALIZE TD3 INSTANCE ================================================##
env = HopperEnv()

# For Reproducibility:
torch.manual_seed(0)
np.random.seed(0)

# Environment action and states
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_action = float(env.action_space.low[0])
min_Val = torch.tensor(1e-7).float().to(device) 

# Exploration Noise
exploration_noise = OrnsteinUhlenbeckNoise(action_dim)


## TRAINING ===============================================================##
def map(value, from_min, from_max, to_min, to_max):
    return np.clip((value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min, to_min, to_max)

def plot_learning_curve(x, scores):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    #plt.savefig(figure_file)
    plt.show()

def train(logging = False):
    
    # Create a DDPG instance
    global agent
    agent = TD3(state_dim, action_dim)
    #print("State dim: {}, Action dim: {}".format(state_dim, action_dim))

    env.reset()
    MAX_TIME_STEPS = int(env.sim_time / env.tn)+1
    
    episodes = []
    best_episode = {'index':None,'value':0}
    score_hist = [] # Initialize the list where all historical rewards of each episode are stored
    prune = False
    
    # Train the agent for the number of episodes set:
    
    for i in range(MAX_EPISODE):
        # prune runs without minimum reward at 50 Episodes
        if i == 25:
            avg_reward = np.mean(score_hist[-10:])
            if avg_reward < 10:
                print('PRUUUUNED! ')
                prune = True
                break

        if i == 45:
            avg_reward = np.mean(score_hist[-10:])
            if avg_reward < 50:
                print('PRUUUUNED! ')
                prune = True
                break
        
        start_time = time.time()
        
        total_reward = 0
        step = 0
        state = env.reset() # [x_target, x, a]
        
        log = np.zeros((8,MAX_TIME_STEPS))
        
        start_time = time.time()
        
        for t in range(MAX_TIME_STEPS):
            
            action = agent.select_action(state) # range [-1..1]
            action = map(action, -1, 1, min_action, max_action) # [0..7]
            
            # # Add Gaussian noise to actions for exploration
            # action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(min_action, max_action)

            # OrnsteinUhlenbeckNoise
            noise = exploration_noise.generate()
            action_with_noise = action + noise
            
            # Clip the action if necessary
            action_with_noise = np.clip(action_with_noise, min_action, max_action)
            
            #action = np.array([7]) # test full throttle

            y, reward, done, info = env.step(action_with_noise,raw=action)
            
            if logging:
                
                log[0,t] = y[1] # x
                log[1,t] = info[1] # v
                log[2,t] = y[2] # a
                log[3,t] = action[0] # p_set
                log[4,t] = info[0] # p_actual
                log[5,t] = y[0] # x_target
                log[6,t] = y[1] - y[0] # error
                log[7,t] = reward # reward
            
    
            total_reward += reward
            # if render and i >= render_interval : env.render()
            agent.replay_buffer.push((state, y, action_with_noise, reward, float(done)))
    
            state = y
    
            if done:
                break
        
        score_hist.append(total_reward)
        if logging:
            episodes.append(log)
    
        sim_time = time.time()
        
        agent.update()
    
        end_time = time.time()
        print(f"Episode: {i} | Total Reward: {total_reward:5.2f} | Simulation {1000*(sim_time-start_time):4.2f} ms | Agend update {1000*(end_time-start_time):4.2f} ms ", end='\r', flush=True)
        
    env.close()
    if logging:
        return (agent,log,score_hist)
    else:
        return (agent, prune)

def test(agent,logging=False):
    print()

    all_test_reward = 0
    test_iteration=100
    ep_r = 0

    env.reset()
    MAX_TIME_STEPS = int(env.sim_time / env.tn)+1

    logs = []

    for i in tqdm(range(test_iteration)):
        state = env.reset()
        log = np.zeros((8,MAX_TIME_STEPS))
        
        for t in range(MAX_TIME_STEPS):

            action = agent.select_action(state)
            action = map(action, -1, 1, min_action, max_action)
            y, reward, done, info = env.step(action)
            #print(len(y))
            
            if logging:
                
                log[0,t] = y[1] # x
                log[1,t] = info[1] # v
                log[2,t] = y[2] # a
                log[3,t] = action[0] # p_set
                log[4,t] = info[0] # p_actual
                log[5,t] = y[0] # x_target
                log[6,t] = y[1] - y[0] # error
                log[7,t] = reward # reward
            
            ep_r += reward

            state = y
            
        all_test_reward += ep_r
        ep_r = 0
        logs.append(log)
            
    score = all_test_reward/test_iteration
    if score > 100:
        print('exceptional :)')
        agent.save(int(score))
        
    print(f'final test score: {score:0.2f}                                                                        ')
    
    env.close()
    if logging:
        return logs
    else:
        return score    

# runs the training with given hyperparameters and returns score of test
def run(params, final=False):
    # update global hyperparameters
    global BUFFER_SIZE, BATCH_SIZE, UPDATE_ITERATION, tau, gamma, H1, H2, learning_rate_actor, learning_rate_critic, policy_freq

    H1 = params['actor_neurons']
    H2 = params['critic_neurons']
    learning_rate_actor = params['lr_actor']
    learning_rate_critic = params['lr_critic']
    tau = params['tau']
    gamma = params['gamma']
    batch_size = params['batch_size']
    
    agent, prune = train()
    if not prune:
        accuracy = test(agent)
    else:
        accuracy = 0
    
    if final:
        agent.save()
        logs = test(agent,logging=True)
        if not os.path.exists('./img/test/'):
            os.makedirs('./img/test/')
        for i in range(10):
            plot_doc(logs[random.randint(0, len(logs)-1)],f'img/test/{i}.png',silent=True)
        print('agent saved.')
        
    return accuracy