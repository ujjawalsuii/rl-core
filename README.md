# 🧠 rl-core: Foundations of Reinforcement Learning

> *"What I cannot create, I do not understand."* — Richard Feynman

**rl-core** is a minimalist, high-performance Reinforcement Learning library built from scratch in pure Python and NumPy. Unlike frameworks that abstract away the mechanics (like Stable Baselines), this library implements the mathematical foundations of RL directly from Sutton & Barto's *Reinforcement Learning: An Introduction*.

### ⚡ Key Features
* **Zero Heavy Dependencies:** No PyTorch, no TensorFlow. Just math.
* **Tile Coding:** Custom implementation of sparse coding for continuous state spaces (Sutton Ch. 9).
* **Actor-Critic Architecture:** Implementation of One-Step Actor-Critic with separate policy/value heads (Sutton Ch. 13).
* **Linear Function Approximation:** Efficient state representation for low-latency inference.

### 🛠️ Architecture
The library treats the Agent and the State Representation as distinct modules:
1.  **TileCoder:** Hashes continuous coordinates $(x, y)$ into a high-dimensional sparse binary vector $\phi(s)$.
2.  **Critic ($w$):** Learns the state-value function $V(s) \approx w^T \phi(s)$ via TD-Error.
3.  **Actor ($\theta$):** Optimizes the policy $\pi(a|s)$ via Policy Gradient ascent.

### 🚀 Usage
```python
from rl_core.agents import ActorCriticAgent
from rl_core.tile_coding import TileCoder

# Initialize Systems
encoder = TileCoder(iht_size=4096, num_tilings=8)
agent = ActorCriticAgent(num_features=4096, num_actions=4)

# Training Loop
features = encoder.get_features(state=(0.5, 0.2))
action = agent.select_action(features)
# ... Environment Step ...
agent.learn(features, action, reward, next_features, done=False)
