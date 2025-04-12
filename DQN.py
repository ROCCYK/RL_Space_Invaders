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
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
BUFFER_SIZE = 100_000
MIN_REPLAY_SIZE = 10_000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1_000_000
TARGET_UPDATE_FREQ = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "dqn_spaceinvaders.pth"

# Frame Preprocessing
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

# DQN Network
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, done, next_state):
        self.buffer.append((state, action, reward, done, next_state))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, dones, next_states = zip(*transitions)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE),
            torch.tensor(actions).to(DEVICE),
            torch.tensor(rewards).to(DEVICE),
            torch.tensor(dones).to(DEVICE),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)

# Epsilon-greedy action
def select_action(state, epsilon, policy_net, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return policy_net(state).argmax(dim=1).item()

# Save/Load
def save_model(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path=MODEL_PATH):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        print(f"Model loaded from {path}")
    else:
        print(f"No saved model found at {path}")

# Plotting

def plot_rewards(rewards, window=10, save_path="rewards_plot.png"):
    plt.figure(figsize=(12, 5))
    plt.plot(rewards, label="Episode Reward")
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode="valid")
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f"{window}-Episode Moving Avg")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN: Space Invaders")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

# Main Training Loop
def train_dqn(env_name="ALE/SpaceInvaders-v5", episodes=500, load_existing=False, render=False, evaluate_only=False, log_csv=True):
    env = gym.make(env_name, render_mode="human" if render else None)
    n_actions = env.action_space.n
    frame_stack = FrameStack(4)

    policy_net = DQN((4, 84, 84), n_actions).to(DEVICE)
    target_net = DQN((4, 84, 84), n_actions).to(DEVICE)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    if load_existing:
        load_model(policy_net)
        target_net.load_state_dict(policy_net.state_dict())

    log_data = []
    best_reward = float('-inf')
    best_episode = -1

    state, _ = env.reset()
    state = frame_stack.reset(state)
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = frame_stack.step(next_state)
        replay_buffer.push(state, action, reward, done, next_state)
        state = frame_stack.reset(env.reset()[0]) if done else next_state

    target_net.load_state_dict(policy_net.state_dict())
    rewards_all = []
    steps_done = 0

    for episode in trange(episodes):
        state = frame_stack.reset(env.reset()[0])
        total_reward = 0
        done = False

        while not done:
            if render:
                env.render()

            epsilon = 0.0 if evaluate_only else max(EPSILON_END, EPSILON_START - steps_done / EPSILON_DECAY)
            action = select_action(state, epsilon, policy_net, n_actions)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = frame_stack.step(next_obs)
            total_reward += reward

            replay_buffer.push(state, action, reward, done, next_state)
            state = next_state

            if not evaluate_only:
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, dones, next_states = batch
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + GAMMA * next_q_values * (~dones)
                loss = nn.MSELoss()(q_values, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if steps_done % TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                steps_done += 1

        rewards_all.append(total_reward)
        epsilon_logged = 0.0 if evaluate_only else max(EPSILON_END, EPSILON_START - steps_done / EPSILON_DECAY)
        log_data.append({"episode": episode, "total_reward": total_reward, "epsilon": epsilon_logged})

        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode
            save_model(policy_net)
            print(f"\U0001f3c6 New Best Reward: {best_reward:.2f} at Episode {best_episode} — Model Saved!")

        print(f"Episode {episode} - Reward: {total_reward:.2f} - Epsilon: {epsilon_logged:.4f}")

    env.close()
    print(f"\n✅ Best Reward: {best_reward:.2f} achieved at Episode {best_episode}")
    plot_rewards(rewards_all)
    if log_csv:
        df = pd.DataFrame(log_data)
        df.to_csv("training_log.csv", index=False)
        print("Log saved to training_log.csv")

    return rewards_all

if __name__ == "__main__":
    # Change parameters as needed
    train_dqn(episodes=500, load_existing=False, render=False, evaluate_only=False, log_csv=True)