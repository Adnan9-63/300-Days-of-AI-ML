# 🐦 Flappy Bird AI Agent — Deep Q-Network (DQN)

An autonomous AI agent that **learns to play Flappy Bird from scratch** using Deep Reinforcement Learning. The agent starts with zero knowledge, explores through random actions, and gradually masters the game — navigating through pipes purely by maximizing reward signals.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL_Environment-green)
![Status](https://img.shields.io/badge/Status-Trained-brightgreen)

---

## 📌 Overview

This project implements a **Deep Q-Network (DQN)** agent that learns to play Flappy Bird through trial-and-error interaction with the game environment. The agent uses the following key techniques from the landmark [DQN paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236):

- **Deep Q-Network** — A neural network approximates the Q-value function over a continuous 12-dimensional state space
- **Experience Replay** — Past transitions are stored and randomly sampled to break temporal correlation and stabilize training
- **Target Network** — A periodically-synced copy of the policy network provides stable Q-value targets during learning
- **Epsilon-Greedy Exploration** — Balances exploration of new strategies with exploitation of learned behavior via a decaying ε schedule

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FlappyBird-v0 Environment            │
│         state (12 features) ↓        ↑ action (0 or 1)  │
└──────────────────────────┬──────────┬───────────────────┘
                           │          │
                    ┌──────▼──────────┴──────┐
                    │        Agent            │
                    │  ┌──────────────────┐   │
                    │  │   Policy DQN     │   │    ε-greedy
                    │  │  12 → 256 → 2   │◄──┼─── action selection
                    │  └────────┬─────────┘   │
                    │           │ optimize     │
                    │  ┌────────▼─────────┐   │
                    │  │   Target DQN     │   │    stable
                    │  │  12 → 256 → 2   │───┼─── Q-value targets
                    │  └──────────────────┘   │
                    │  ┌──────────────────┐   │
                    │  │ Experience Replay │   │    random
                    │  │   Buffer (100K)  │───┼─── mini-batch sampling
                    │  └──────────────────┘   │
                    └─────────────────────────┘
```

### Neural Network (DQN)

| Layer | Type | Shape | Activation |
|-------|------|-------|------------|
| Input | — | 12 features | — |
| Hidden | Fully Connected | 12 → 256 | ReLU |
| Output | Fully Connected | 256 → 2 | None (raw Q-values) |

**Input** (12 features): Bird Y-position, velocity, pipe distances, pipe positions, and other spatial features.  
**Output** (2 Q-values): Expected return for each action — `0` (no flap) and `1` (flap).

---

## ⚙️ How It Works

### The Training Loop

1. **Observe** the current state (12 float features from the environment)
2. **Select an action** using ε-greedy policy:
   - With probability ε → random action (exploration)
   - With probability 1−ε → action with highest Q-value (exploitation)
3. **Execute** the action, receive reward and next state
4. **Store** the transition `(s, a, r, s', done)` in the replay buffer
5. **Sample** a random mini-batch of 32 transitions from the buffer
6. **Compute loss** using the Bellman equation:

$$L = \frac{1}{N}\sum_{i}\left(Q_{\text{policy}}(s_i, a_i) - \left[r_i + \gamma \cdot (1 - \text{done}_i) \cdot \max_{a'} Q_{\text{target}}(s'_i, a')\right]\right)^2$$

7. **Update** the policy network via backpropagation (Adam optimizer)
8. **Sync** the target network every 10 steps

### Epsilon Decay Schedule

| Phase | Episodes | Epsilon (ε) | Behavior |
|-------|----------|-------------|----------|
| Early | 0–1000 | 1.0 → 0.61 | Mostly random exploration |
| Mid | 1000–5000 | 0.61 → 0.08 | Gradually exploiting learned policy |
| Late | 5000+ | 0.05 (floor) | Mostly exploiting, 5% random exploration |

---

## 📊 Training Results

The agent's best reward progression over 20,000+ episodes of training:

| Episode | Best Reward | Milestone |
|---------|-------------|-----------|
| 1 | -6.30 | Random play — dies immediately |
| 46 | -3.30 | Learning basic survival |
| 553 | -0.90 | Nearly breaking even |
| 1,537 | +0.30 | First pipe cleared! |
| 1,816 | +1.50 | Passing multiple pipes consistently |
| 2,326 | +2.70 | Developing strong navigation skills |
| 3,845 | +5.20 | Clearing 5+ pipes per run |
| **5,740** | **+6.80** ✅ | **Peak performance — 6+ pipes cleared!** |

> The agent evolves from dying instantly (reward -6.3) to clearing 6+ pipes per game (reward +6.8) — all through self-play and reward optimization, with no human-programmed strategy.

---

## 📂 Project Structure

```
flappy_bird_agent/
├── agent.py                 # Training & testing orchestrator (main entry point)
├── dqn.py                   # Deep Q-Network architecture (PyTorch)
├── experience_replay.py     # Replay memory buffer (deque-based FIFO)
├── game_flappy-bird.py      # Manual play script (play with keyboard)
├── parameters.yaml          # Hyperparameter configuration
├── architecture.md          # Architecture planning notes
├── runs/
│   ├── flappybirdv0.pt      # Saved model weights (best policy)
│   └── flappybirdv0.log     # Training log (best rewards per episode)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/flappy-bird-dqn.git
cd flappy-bird-dqn

# Install dependencies
pip install gymnasium flappy-bird-gymnasium torch pyyaml pygame
```

### Usage

```bash
# Train the agent (runs indefinitely — Ctrl+C to stop)
python agent.py flappybirdv0 --train

# Watch the trained agent play (loads saved model)
python agent.py flappybirdv0

# Play the game manually with spacebar
python game_flappy-bird.py
```

---

## 🔧 Hyperparameters

All hyperparameters are configured in [`parameters.yaml`](parameters.yaml):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` | 0.001 | Learning rate (Adam optimizer) |
| `gamma` | 0.99 | Discount factor for future rewards |
| `epsilon_init` | 1.0 | Initial exploration rate |
| `epsilon_min` | 0.05 | Minimum exploration rate |
| `epsilon_decay` | 0.9995 | Multiplicative decay per episode |
| `replay_memory_size` | 100,000 | Max transitions in replay buffer |
| `mini_batch_size` | 32 | Training batch size |
| `network_sync_rate` | 10 | Steps between target network syncs |
| `reward_threshold` | 1,000 | Episode reward cap |

---

## 🧠 Key Concepts

| Concept | Role in This Project |
|---------|---------------------|
| **Deep Q-Network** | Neural network approximates Q(s, a) for continuous state spaces |
| **Experience Replay** | Breaks correlation in training data by random sampling from a buffer |
| **Target Network** | Stabilizes training by providing fixed Q-value targets |
| **Epsilon-Greedy** | Balances exploration (random) vs. exploitation (learned policy) |
| **Bellman Equation** | Defines the optimal Q-value recursively: Q = r + γ·max Q' |
| **MSE Loss** | Measures gap between predicted and target Q-values |
| **Adam Optimizer** | Adaptive learning rate optimizer for gradient descent |

---

## 🔮 Potential Improvements

- [ ] **Double DQN** — Use policy network to select actions, target network to evaluate (reduces Q-value overestimation)
- [ ] **Dueling DQN** — Separate state-value and advantage streams for better state evaluation
- [ ] **Prioritized Experience Replay** — Sample important (high TD-error) transitions more frequently
- [ ] **CNN on Raw Pixels** — Replace the 12-feature state with raw screen frames for end-to-end visual learning
- [ ] **Soft Target Updates** — Polyak averaging instead of hard weight copying for smoother target transitions
- [ ] **TensorBoard Integration** — Real-time training metrics visualization (loss, reward, epsilon curves)

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **PyTorch** | Deep learning framework (neural networks, GPU acceleration) |
| **Gymnasium** | Standardized RL environment API |
| **Flappy Bird Gymnasium** | Flappy Bird wrapped as a Gymnasium environment |
| **PyGame** | Game rendering and keyboard input |
| **PyYAML** | Configuration file parsing |

---

## 📚 References

- Mnih, V., et al. (2015). [*Human-level control through deep reinforcement learning*](https://www.nature.com/articles/nature14236). Nature, 518(7540), 529–533.
- Sutton, R. S., & Barto, A. G. (2018). [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book.html). MIT Press.
- [Gymnasium Documentation](https://gymnasium.faraday.ai/)
- [Flappy Bird Gymnasium](https://github.com/markub3327/flappy-bird-gymnasium)

---

## 📄 License

This project is for educational and portfolio purposes.
