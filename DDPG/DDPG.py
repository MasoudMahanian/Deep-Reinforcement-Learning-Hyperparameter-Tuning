import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gymnasium as gym
from noise import OrnsteinUhlenbeckNoise

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,max_action,hidden_size,more_layer=True):
        super(Actor,self).__init__()
        self.more_layer = more_layer
        self.fc1 = nn.Linear(state_dim,hidden_size)
        if self.more_layer:
            self.fc2 = nn.Linear(hidden_size,hidden_size)
            self.fc3 = nn.Linear(hidden_size,action_dim)
        else:
            self.fc2 = nn.Linear(hidden_size,action_dim)
        self.max_action = max_action
    def forward(self,state):
        x = torch.relu(self.fc1(state))
        if self.more_layer:
            x = torch.relu(self.fc2(x))
            x = self.max_action*torch.tanh(self.fc3(x))
        else:
            x = self.max_action*torch.tanh(self.fc2(x))
        return x
    
class Critic(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_size,more_layer=True):
        super(Critic,self).__init__()
        self.more_layer=more_layer
        self.fc1 = nn.Linear(state_dim+action_dim,hidden_size)
        if self.more_layer:
            self.fc2 = nn.Linear(hidden_size,hidden_size)
            self.fc3 = nn.Linear(hidden_size,1)
        else :
            self.fc2 = nn.Linear(hidden_size,1)
        
    def forward(self,state,action):
        x = torch.cat([state,action],dim=1)
        x = torch.relu(self.fc1(x))
        if self.more_layer:
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = self.fc2(x)
        return x

class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self,state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self,batch_size):
        batch = random.sample(self.buffer,batch_size)
        state, action, reward, next_state, done = map(np.array,zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
    
class DDPG:
    def __init__(self,state_dim,action_dim,max_action,hidden_size,
                 learning_rate_actor=1e-4,learning_rate_critic=1e-3,
                 gamma=0.99,tau=0.05,
                 more_layer=True):
        self.actor = Actor(state_dim, action_dim,max_action,hidden_size,more_layer)
        self.target_actor = Actor(state_dim, action_dim,max_action,hidden_size,more_layer)
        self.critic = Critic(state_dim,action_dim,hidden_size,more_layer)
        self.target_critic = Critic(state_dim,action_dim,hidden_size,more_layer)


        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=learning_rate_critic)

        self.reply_buffer = ReplayBuffer(capacity=1_000_000)
        self.max_action = max_action

        self.gamma = gamma
        self.tau = tau

        self.noise = OrnsteinUhlenbeckNoise(action_dim=action_dim)
    

    def select_action(self,state, add_noise = True):
        # state = torch.FloatTensor(state).reshape(1,-1)
        state = torch.FloatTensor(state.reshape(1,-1))
        # no computation Graph
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        if add_noise:
            noise = self.noise.sample()
            action = noise + action
        
        action = np.clip(action,-self.max_action,self.max_action)
        return action
    

    def train(self,batch_size=64):
        if len(self.reply_buffer)<batch_size:
            return
        state, action, reward, next_state, done = self.reply_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).reshape(-1,1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).reshape(-1,1)



        # what we want?: Q_target
        with torch.no_grad():            
            target_action = self.target_actor(next_state)
            target_q = self.target_critic(next_state, target_action) # we want q but target
            target_q = reward + (1-done)*self.gamma*target_q

        # update critic
        current_q = self.critic(state, action) # we want q but use current S and A => Q
        critic_loss = nn.MSELoss()(current_q,target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #update Actor
        actor_loss = -self.critic(state,self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # now target
        for param, target_param in zip(self.actor.parameters(),self.target_actor.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

        for param, target_param in zip(self.critic.parameters(),self.target_critic.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
        
def train_ddpg(env_name = "Pendulum-v1", episodes = 200, max_step =500,
               hidden_size=256, learning_rate_actor=1e-4,learning_rate_critic=1e-3,
               batch_size=128, gamma=0.99,tau=0.005,
               more_layer=True):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    agent = DDPG(state_dim=state_dim,
                 action_dim=action_dim,
                 max_action=max_action,
                 hidden_size=hidden_size,
                 learning_rate_actor=learning_rate_actor,
                 learning_rate_critic=learning_rate_critic,
                 gamma=gamma,
                 tau=tau,
                 more_layer=more_layer
                 )
    reward_for_run = []
    total_steps = 0
    for ep in range(episodes):
        state , _ = env.reset()
        agent.noise.reset()
        episode_reward = 0
        for step in range(max_step):
            # if total_steps <4000:
            #     action = env.action_space.sample()
            # else:
            #     action = agent.select_action(state, add_noise=True)
            action = agent.select_action(state, add_noise=True)
            next_state, reward, terminated, truncated,_ =env.step(action)
            done = terminated or truncated
            agent.reply_buffer.push(state, action, reward, next_state, done)
            agent.train(batch_size=batch_size)
            if len(agent.reply_buffer)>1000:
                agent.train(batch_size=batch_size)

            state=next_state
            episode_reward+=reward

            if done:
                break
        reward_for_run.append(episode_reward)
        avg_reward = np.mean(reward_for_run[-10:]) if len(reward_for_run) >= 10 else np.mean(reward_for_run)
        print(f"Episode {ep+1}/{episodes} | Reward: {episode_reward:.2f} | Avg(10): {avg_reward:.2f}")
    
    env.close()
    return reward_for_run

# train_ddpg(episodes=1)
# quit()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_results(all_rewards, param_grid):
    """
    رسم نمودارهای مختلف برای تحلیل نتایج
    """
    # تنظیم استایل
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. نمودار منحنی یادگیری برای هر کانفیگ
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # رسم منحنی‌های یادگیری
    for i, (hidden, more_) in enumerate(product(param_grid['hidden_size'], param_grid['more_layer'])):
        label = f"Hidden={hidden}, MoreLayer={more_}"
        
        # میانگین و انحراف معیار برای هر اپیزود
        rewards = all_rewards[i]  # shape: (n_runs, n_episodes)
        mean_rewards = np.mean(rewards, axis=0)
        std_rewards = np.std(rewards, axis=0)
        
        # محاسبه moving average برای صاف کردن منحنی
        window = 10
        if len(mean_rewards) >= window:
            mean_smooth = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
            std_smooth = np.convolve(std_rewards, np.ones(window)/window, mode='valid')
            x_axis = np.arange(window-1, len(mean_rewards))
        else:
            mean_smooth = mean_rewards
            std_smooth = std_rewards
            x_axis = np.arange(len(mean_rewards))
        
        # رسم منحنی با ناحیه اطمینان
        axes[0].plot(x_axis, mean_smooth, label=label, linewidth=2)
        axes[0].fill_between(x_axis, 
                            mean_smooth - std_smooth, 
                            mean_smooth + std_smooth, 
                            alpha=0.2)
    
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Average Reward', fontsize=12)
    axes[0].set_title('Learning Curves with Confidence Intervals', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. نمودار Boxplot برای مقایسه نهایی
    final_rewards = []
    labels = []
    
    for i, (hidden, more_) in enumerate(product(param_grid['hidden_size'], param_grid['more_layer'])):
        rewards = all_rewards[i]
        # میانگین 10 اپیزود آخر برای هر run
        final_avg = np.mean(rewards[:, -10:], axis=1)
        final_rewards.extend(final_avg)
        labels.extend([f"H={hidden}\nML={more_}"] * len(final_avg))
    
    df = pd.DataFrame({'Configuration': labels, 'Final Reward': final_rewards})
    sns.boxplot(x='Configuration', y='Final Reward', data=df, ax=axes[1])
    axes[1].set_title('Final Performance Comparison (Last 10 Episodes)', fontsize=14)
    axes[1].set_ylabel('Average Reward', fontsize=12)
    axes[1].set_xlabel('Configuration', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 3. نمودار اضافی: مقایسه عملکرد در طول زمان
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (hidden, more_) in enumerate(product(param_grid['hidden_size'], param_grid['more_layer'])):
        rewards = all_rewards[i]
        mean_rewards = np.mean(rewards, axis=0)
        std_rewards = np.std(rewards, axis=0)
        
        x_axis = np.arange(len(mean_rewards))
        ax.plot(x_axis, mean_rewards, label=f"H={hidden}, ML={more_}", linewidth=2)
        ax.fill_between(x_axis, 
                       mean_rewards - std_rewards, 
                       mean_rewards + std_rewards, 
                       alpha=0.2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Performance Comparison with Standard Deviation', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 4. چاپ آمار نهایی
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    
    for i, (hidden, more_) in enumerate(product(param_grid['hidden_size'], param_grid['more_layer'])):
        rewards = all_rewards[i]
        final_avg = np.mean(rewards[:, -10:], axis=1)
        
        print(f"\nConfiguration: Hidden={hidden}, MoreLayer={more_}")
        print(f"  Mean Final Reward: {np.mean(final_avg):.2f} ± {np.std(final_avg):.2f}")
        print(f"  Best Final Reward: {np.max(final_avg):.2f}")
        print(f"  Worst Final Reward: {np.min(final_avg):.2f}")
        
        # بهترین عملکرد در کل
        best_episode_reward = np.max(rewards)
        best_episode = np.argmax(rewards)
        print(f"  Best Episode Reward: {best_episode_reward:.2f} (Episode {best_episode})")
    
    # 5. رسم هیستوگرام توزیع پاداش‌ها
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (hidden, more_) in enumerate(product(param_grid['hidden_size'], param_grid['more_layer'])):
        rewards = all_rewards[i]
        final_avg = np.mean(rewards[:, -10:], axis=1)
        ax.hist(final_avg, bins=10, alpha=0.5, label=f"H={hidden}, ML={more_}")
    
    ax.set_xlabel('Average Final Reward', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Final Performance', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def test_agent(params):
    hidden_size = params['hidden_size']
    more_layer = params['more_layer']

    print(f"\n=== Testing: more_layer={more_layer}, 1idden={hidden_size}===")

    reward = train_ddpg(env_name = "Pendulum-v1", episodes = 500, max_step =500,
               hidden_size=hidden_size, learning_rate_actor=1e-4,learning_rate_critic=1e-3,
               batch_size=128, gamma=0.99,tau=0.005,
               more_layer=more_layer)

    

    return reward


from itertools import product

param_grid = {
    'hidden_size': [256,128],
    'more_layer' : [True,False]
    # 'more_layer' : [True]

}

all_rewards_for_all_runs=[]

for hidden,more_ in product(
    param_grid['hidden_size'],
    param_grid['more_layer']
):
    params = {
        'hidden_size': hidden,
        'more_layer' : more_
    }
    all_rewards_for_same_params = []
    for i in range(1):
        rewards = test_agent(params)
        all_rewards_for_same_params.append(rewards)

    all_rewards_for_all_runs.append(all_rewards_for_same_params)

all_rewards = np.array(all_rewards_for_all_runs)



plot_results(all_rewards, param_grid)