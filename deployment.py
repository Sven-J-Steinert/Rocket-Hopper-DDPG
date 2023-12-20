import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

x_offset = 0.11 # [m] position offset from zero point

# init NN computation
cuda = torch.cuda.is_available() #check for CUDA
device = torch.device("cuda" if cuda else "cpu")
print("Job will run on {}".format(device))

def map(value, from_min, from_max, to_min, to_max):
    return np.clip((value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min, to_min, to_max)

max_bar = 7 # (?) 7 in simulation action space 

# map p_actual uint12 value ([0..4095] to [bar]
def val_2_bar(val):
    return map(val, 0, 4095, 0, max_bar) # [0..4095] to [0..max_bar]

# map p_set [bar] to uint12 value [0..4095]
def bar_2_val(bar):
    return int(map(bar, 0, max_bar, 0, 4095)) # [0..max_bar] to [0..4095]

def RX(payload):
    # incoming message from MCU - string expected
    # "< 74282: 9.73: 1.21: 1980: 0: 0 >"

    content = payload[1:-1].split(':') # remove "<" and ">" and split
    for index, val in enumerate(content):
        content[index] = float(val) # convert string to float

    # content is [74282.0, 9.73, 1.21, 1980.0, 0.0, 0.0]
    
    x = content[2] - x_offset # position [m]
    a = content[1] # acceleration [m/s²]
    p_actual = val_2_bar(content[3]) # pressure [bar]
        
    return (x,a,p_actual)

def TX(p_set,timer=None):
    # craft send message
    # < 39872: 2109 >
        
    msg = f'< {timer}: {p_set}>'
    return msg

# =============================================================================

# define DDPG class
directory = './'

H1=32  # Neuron of 1st Layer
H2=H1*2  # Neurons of 2nd layer

learning_rate_actor = 0.0 # not relevant
learning_rate_critic = 0.0 # not relevant

class DDPG(object):
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
        print(f"learning rates: actor {learning_rate_actor} | critic {learning_rate_critic}")
        
        self.actor = Actor(state_dim, action_dim, H1).to(device)
        self.actor_target = Actor(state_dim, action_dim,  H1).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)

        self.critic = Critic(state_dim, action_dim,  H2).to(device)
        self.critic_target = Critic(state_dim, action_dim,  H2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate_critic)
        # learning rate

        

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
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss as the negative mean Q value using the critic network and the actor network
            actor_loss = -self.critic(state, self.actor(state)).mean()
            

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
    def save(self):
        """
        Saves the state dictionaries of the actor and critic networks to files
        """
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        

    def load(self):
        """
        Loads the state dictionaries of the actor and critic networks to files
        """
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))

# define Replaybuffer
BUFFER_SIZE=1_000_000 # Buffer size of 1 million entries

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

# define Actor and Critic
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
            nn.Linear(H1, action_dim)
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
        self.net = nn.Sequential(
            nn.Linear(n_states + action_dim, H2), 
            nn.ReLU(), 
            nn.Linear(H2, H2), 
            nn.ReLU(), 
            nn.Linear(H2, H2), 
            nn.ReLU(), 
            nn.Linear(H2, action_dim)
        )
        
    def forward(self, state, action):
        return self.net(torch.cat((state, action), 1))


