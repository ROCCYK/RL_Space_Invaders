import pandas as pd
import matplotlib.pyplot as plt

log_df = pd.read_csv("ppo_training_log.csv")

plt.figure(figsize=(12, 5))
plt.plot(log_df["episode"], log_df["total_reward"], label="Episode Reward")

# Moving average
window = 10
moving_avg = log_df["total_reward"].rolling(window).mean()
plt.plot(log_df["episode"], moving_avg, label=f"{window}-Episode Moving Avg")

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Learning Curve - Space Invaders")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("ppo_rewards_plot.png")
plt.show()
