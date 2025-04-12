# 🎮 RL_Space_Invaders

An interactive reinforcement learning project where agents trained using **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)** play Atari's classic **Space Invaders**. The project includes a Streamlit app to visualize and compare the performance of both agents.

## 🚀 Features

- **DQN and PPO Implementations**: Custom-built agents trained to play Space Invaders.
- **Streamlit Interface**: Visualize agent gameplay and compare performances.
- **Benchmarking**: Run multiple simulations to compare average rewards between DQN and PPO agents.
- **Pretrained Models**: Includes pretrained weights for immediate demonstrations.

## 🧠 Algorithms

### Deep Q-Network (DQN)

A value-based method where a neural network approximates the Q-value function, guiding the agent to take actions that maximize expected rewards.

### Proximal Policy Optimization (PPO)

A policy-gradient method that optimizes the policy directly, ensuring stable and efficient learning by limiting policy updates.

## 📦 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ROCCYK/RL_Space_Invaders.git
   cd RL_Space_Invaders
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   *Note*: Ensure you have a working installation of PyTorch compatible with your system. Visit [PyTorch's official site](https://pytorch.org/get-started/locally/) for installation instructions tailored to your setup.

## 🎮 Running the Streamlit App

🚀 Try the live demo here: [rl-space-invaders.streamlit.app](https://rl-space-invaders.streamlit.app/)

Launch the Streamlit application to visualize agent gameplay:

```bash
streamlit run streamlitapp.py
```

### App Features

- **Agent Selection**: Choose between DQN and PPO agents.
- **Run Agent**: Watch the selected agent play Space Invaders in real-time.
- **Benchmarking**: Run 100 simulations for each agent and compare their average rewards.

## 📁 Repository Structure

```
RL_Space_Invaders/
├── DQN.py                 # DQN agent implementation
├── PPO.py                 # PPO agent implementation
├── dqn_spaceinvaders.pth  # Pretrained DQN model weights
├── ppo_spaceinvaders.pth  # Pretrained PPO model weights
├── streamlitapp.py        # Streamlit application
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 📝 Acknowledgments

- **OpenAI Gymnasium**: For providing the Space Invaders environment.
- **PyTorch**: For building and training neural network models.
- **Streamlit**: For creating an interactive web interface.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
