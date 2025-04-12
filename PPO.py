import gymnasium as gym
import numpy as np
import random
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
import ale_py

# Hyperparameters
GAMMA = 0.99
LR = 2.5e-4
CLIP_EPS = 0.1
BATCH_SIZE = 32
UPDATE_EPOCHS = 4
ENTROPY_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
ROLLOUT_STEPS = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ppo_spaceinvaders.pth"

# Frame preprocessing
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    return frame

class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self, frame):
        processed = preprocess(frame)
        for _ in range(self.k):
            self.frames.append(processed)
        return np.stack(self.frames, axis=0)

    def step(self, frame):
        processed = preprocess(frame)
        self.frames.append(processed)
        return np.stack(self.frames, axis=0)

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return self.actor(x), self.critic(x)

# Rollout Buffer for PPO
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, state, action, logprob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value, gamma=GAMMA, lam=0.95):
        values = self.values + [last_value]
        gae = 0
        returns = []
        advantages = []
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        self.returns = returns
        self.advantages = advantages

    def get(self):
        return map(lambda x: torch.tensor(x, dtype=torch.float32).to(DEVICE), 
                   [self.states, self.actions, self.logprobs, self.returns, self.advantages])

    def clear(self):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

# Action selection with logprob
def select_action(model, state):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    logits, value = model(state_t)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    return action.item(), dist.log_prob(action).item(), value.item()

# Training loop
def train_ppo(env_name="ALE/SpaceInvaders-v5", episodes=500):
    env = gym.make(env_name)
    n_actions = env.action_space.n
    frame_stack = FrameStack(4)
    model = ActorCritic((4, 84, 84), n_actions).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    log_data = []
    best_reward = float('-inf')
    buffer = RolloutBuffer()

    for episode in trange(episodes):
        state = frame_stack.reset(env.reset()[0])
        total_reward = 0
        done = False
        buffer.clear()

        for _ in range(ROLLOUT_STEPS):
            action, logprob, value = select_action(model, state)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = frame_stack.step(obs)

            buffer.add(state, action, logprob, reward, done, value)
            state = frame_stack.reset(env.reset()[0]) if done else next_state
            total_reward += reward

            if done:
                state = frame_stack.reset(env.reset()[0])

        _, _, last_val = select_action(model, state)
        buffer.compute_returns_and_advantages(last_val)

        states, actions, logprobs_old, returns, advantages = buffer.get()
        actions = actions.long()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(UPDATE_EPOCHS):
            idx = np.random.permutation(len(states))
            for start in range(0, len(states), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_idx = idx[start:end]

                logits, values = model(states[batch_idx])
                dist = torch.distributions.Categorical(logits=logits)
                entropy = dist.entropy().mean()
                new_logprobs = dist.log_prob(actions[batch_idx])
                ratio = (new_logprobs - logprobs_old[batch_idx]).exp()

                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages[batch_idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values.squeeze(), returns[batch_idx])
                loss = actor_loss + VF_COEF * critic_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        log_data.append({"episode": episode, "total_reward": total_reward})
        print(f"Episode {episode} - Reward: {total_reward:.2f}")

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"\U0001f3c6 New Best Reward: {best_reward:.2f} — Model Saved!")

    env.close()
    df = pd.DataFrame(log_data)
    df.to_csv("ppo_training_log.csv", index=False)
    print("Training complete. Log saved to ppo_training_log.csv")

if __name__ == "__main__":
    train_ppo()
