import streamlit as st
import torch
import cv2
import numpy as np
import gymnasium as gym
from collections import deque
import time
import os
import ale_py

# Import DQN components
from DQN import DQN, preprocess, FrameStack, DEVICE
# Import PPO model
from PPO import ActorCritic

DQN_MODEL_PATH = "dqn_spaceinvaders.pth"
PPO_MODEL_PATH = "ppo_spaceinvaders.pth"

# Streamlit UI Setup
st.title("🎮 Space Invaders - DQN vs PPO Demo")
st.markdown("""
This app lets you run a trained agent (DQN or PPO) to play Space Invaders in real-time.

- **DQN**: Uses value-based greedy policy  
- **PPO**: Uses policy gradient and actor-critic structure  

Due to Streamlit's rendering requirements, the OpenAI Gym environment is set to 'rgb_array'
mode instead of 'human'. As a result, some dynamic visual elements like enemy missiles 
may not be visible in the rendered frames, since they are often rendered in intermediate or 
rapidly updating frames that are missed in this display mode.
""")

# Environment Setup
env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
n_actions = env.action_space.n
frame_stack = FrameStack(4)

# Agent selection
agent_type = st.selectbox("Select Agent", ["DQN", "PPO"])

# Model Loading
def load_model(agent_type):
    if agent_type == "DQN":
        model = DQN((4, 84, 84), n_actions).to(DEVICE)
        path = DQN_MODEL_PATH
    else:
        model = ActorCritic((4, 84, 84), n_actions).to(DEVICE)
        path = PPO_MODEL_PATH

    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        return model
    else:
        st.error(f"Model not found: {path}")
        st.stop()

# Action selection
def select_action_dqn(model, state):
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return model(state_tensor).argmax(dim=1).item()

def select_action_ppo(model, state):
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        logits, _ = model(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().item()

# Run a single episode
def run_episode(model, agent_type, render=False):
    state, _ = env.reset()
    state = frame_stack.reset(state)
    total_reward = 0
    done = False

    while not done:
        if agent_type == "DQN":
            action = select_action_dqn(model, state)
        else:
            action = select_action_ppo(model, state)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = frame_stack.step(obs)
        total_reward += reward

        if render:
            frame = env.render()
            stframe.image(frame, caption=f"Reward: {total_reward:.0f}", channels="RGB", use_container_width=True)
            time.sleep(0.03)

    return total_reward

# 🎮 Button: Run Agent Visually
if st.button("▶️ Run Agent"):
    model = load_model(agent_type)
    stframe = st.empty()
    total_reward = run_episode(model, agent_type, render=True)
    st.success(f"🎉 Game Over! Total Reward: {total_reward:.0f}")

# 📊 Button: Benchmark Both Models
if st.button("📊 Run 100 Games & Compare DQN vs PPO"):
    st.info("Running 100 games for each agent... Please wait ⏳")

    dqn_model = load_model("DQN")
    ppo_model = load_model("PPO")

    dqn_scores = []
    ppo_scores = []

    for _ in range(100):
        dqn_scores.append(run_episode(dqn_model, "DQN"))
        ppo_scores.append(run_episode(ppo_model, "PPO"))

    avg_dqn = np.mean(dqn_scores)
    avg_ppo = np.mean(ppo_scores)

    st.subheader("🏁 Results after 100 games")
    st.write(f"**DQN Average Reward:** {avg_dqn:.2f}")
    st.write(f"**PPO Average Reward:** {avg_ppo:.2f}")

    if avg_dqn > avg_ppo:
        st.success("✅ **DQN performed better on average!**")
    elif avg_ppo > avg_dqn:
        st.success("✅ **PPO performed better on average!**")
    else:
        st.warning("⚖️ Both agents performed equally!")
