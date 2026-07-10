import numpy as np
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# DQN Neural network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_size = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Agent DQN
class DQNAgent:
    def __init__(self, state_dim, action_dim,hidden_size=128,learning_rate=0.01,gamma=0.99,batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = DQN(state_dim, action_dim,hidden_size)
        self.target_network = DQN(state_dim, action_dim,hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = batch_size
        
    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Train
class Run():
    def __init__(self):
        pass
    def run(self,hidden_size,learning_rate,gamma,batch_size):
        env = gym.make('CartPole-v1')
        agent = DQNAgent(env.observation_space.shape[0],
                          env.action_space.n,
                         hidden_size=hidden_size,
                         learning_rate=learning_rate,
                         gamma=gamma,
                         batch_size=batch_size)
        episodes = 400
        rewards = []
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
    
            while True:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            rewards.append(total_reward)
            if episode % 10 == 0:
                agent.update_target()
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        env.close()
        print("run is finished")
        avg_reward_last20 = np.mean(rewards[-20:])
        return avg_reward_last20, rewards



import numpy as np
from itertools import product


# Function to test each combination
def test_agent(params):
    hidden_size = params['hidden_size']
    learning_rate = params['learning_rate']
    gamma= params['gamma']
    batch_size = params['batch_size']
    
    print(f"\n=== Testing: LR={learning_rate}, gamma={gamma}, batch={batch_size}, hidden={hidden_size}===")

    runner = Run()
    
    avg_score,reward = runner.run(hidden_size=hidden_size,
                                  learning_rate=learning_rate,
                                  gamma=gamma,
                                  batch_size=batch_size)

    return avg_score,reward

# Check all number combinations.
best_params = None
best_score = -np.inf

# Parameters you want to test
param_grid = {
    'learning_rate': [0.001, 0.01,0.0001],
    'gamma': [0.95,0.99],
    'batch_size': [32,64],
    'hidden_size': [64,128]
}

all_rewards_for_all=[]
for lr, gamma, batch, hidden in product(
    param_grid['learning_rate'],
    param_grid['gamma'],
    param_grid['batch_size'],
    param_grid['hidden_size']
):
    params = {
        'learning_rate': lr,
        'gamma': gamma,
        'batch_size': batch,
        'hidden_size': hidden
    }
    all_rewards_for_same_params = []
    for i in range(3):
        score,rewards = test_agent(params)
        print(f"Test number: {i+1}-> Score: {score}")
        all_rewards_for_same_params.append(rewards)
    if score > best_score:
        best_score = score
        best_params = params
    all_rewards_for_all.append(all_rewards_for_same_params)

all_rewards = np.array(all_rewards_for_all)
# print(all_rewards)
# print(all_rewards.shape[0])
# print(all_rewards[0])
# print(f"\n best param: {best_params}")
# print(f"best_score: {best_score}")
import matplotlib.pyplot as plt
# for RW in all_rewards:
#     mean_rw = np.mean(RW,axis=0)
#     std_rw = np.std(RW,axis=0)
#     # plt.figure(figsize=(12,6))
#     window = 3
#     mean_smooth = np.convolve(mean_rw,np.ones(window)/window,mode='valid')
#     std_smooth = np.zeros_like(mean_smooth)
#     for i in range(len(mean_smooth)):
#         indices = range(i,i+window)
#         std_smooth[i]=np.mean([np.std(RW[:,idx]) for idx in indices])
#     episodes = np.arange(len(mean_smooth))

#     plt.plot(episodes, mean_smooth, linewidth=2, label='mean of (20 tun )')
#     plt.fill_between(episodes, 
#                     mean_smooth - std_smooth, 
#                     mean_smooth + std_smooth, 
#                     alpha=0.3, label='±1 Standard Deviation  ')

# show_legend =[]

# for lr, gamma, batch, hidden in product(
#         param_grid['learning_rate'],
#         param_grid['gamma'],
#         param_grid['batch_size'],
#         param_grid['hidden_size']
# ):
    
#     mani_text = 'lr: '+str(lr)+', gamma: '+str(gamma)+ ', b_size: '+str(batch)+', h_size: '+str(hidden)

#     text1 = "mean (20 run) "+mani_text
#     text2 = "±1 Standard Deviation  of "mani_text
#     show_legend.append(text1)
#     show_legend.append(text2)
# plt.legend(show_legend)

# plt.show()


# Calculating the final score for each combination
final_scores = []
for idx in range(len(all_rewards_for_all)):
    all_runs = all_rewards_for_all[idx]
    avg_score = np.mean([run[-1] for run in all_runs])
    final_scores.append(avg_score)

#2. Find the top 5 combinations
top_n = 5
top_indices = np.argsort(final_scores)[-top_n:][::-1]

# 3. plot
plt.figure(figsize=(12, 6))

for rank, idx in enumerate(top_indices):
    all_runs = all_rewards_for_all[idx]
    mean_rw = np.mean(all_runs, axis=0)
    std_rw = np.std(all_runs, axis=0)
    

    params = list(product(
        param_grid['learning_rate'],
        param_grid['gamma'],
        param_grid['batch_size'],
        param_grid['hidden_size']
    ))[idx]
    
    lr, gamma, batch, hidden = params
    label = f"#{rank+1}: LR={lr}, γ={gamma}, hidden_size={hidden}"
    
    plt.plot(mean_rw, linewidth=2.5, label=label)
    plt.fill_between(range(len(mean_rw)), mean_rw - std_rw, mean_rw + std_rw, alpha=0.2)

plt.xlabel('Episode', fontsize=12)
plt.ylabel('Total Reward', fontsize=12)
plt.title(f'Top {top_n} Hyperparameter Combinations', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()