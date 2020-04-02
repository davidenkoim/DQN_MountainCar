from gym import make
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque, namedtuple
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy

N_STEP = 5
GAMMA = 0.96
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 5
N_EPISODES = 200
DEVICE = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")


def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.0))) / np.array((1.8, 0.07))
    result = []
    result.extend(state)
    return np.array(result)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA ** N_STEP
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )  # Torch model

        def init_weights(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

        self.model.apply(init_weights)
        self.model.to(DEVICE)
        self.target_model = deepcopy(self.model).to(DEVICE)
        self.target_model.eval()
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        states = torch.tensor(batch.state).float().to(DEVICE)
        actions = torch.tensor(batch.action).to(DEVICE)
        rewards = torch.tensor(batch.reward).to(DEVICE)
        next_states = torch.tensor(batch.next_state).float().to(DEVICE)
        non_done = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=DEVICE)

        q = self.model(states).gather(1, actions.reshape((-1,1)))
        target_q = torch.zeros(BATCH_SIZE, device=DEVICE)
        target_q[non_done] = self.gamma * self.target_model(next_states[non_done]).max(1)[0].detach()
        target_q += rewards

        loss = F.smooth_l1_loss(q, target_q.unsqueeze(1))
        self.opt.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt.step()

    def act(self, state):
        state = torch.tensor(state).float().to(DEVICE)
        with torch.no_grad():
            return self.model(state).argmax().item()

    def save(self, path="agent.pkl"):
        torch.save(self.model, path)


def avr_reward(aql, env, n=20):
    avr_r = 0
    for _ in range(n):
        state = env.reset()
        state = transform_state(state)
        reward = 0
        done = False
        while not done:
            action = aql.act(state)
            state, r, done, _ = env.step(action.item())
            state = transform_state(state)
            reward += r
        avr_r += reward
    return avr_r / n


if __name__ == "__main__":
    env = make("MountainCar-v0")
    dqn = DQN(state_dim=2, action_dim=3)
    eps = 0.9
    min_eps = 0.01
    de = 0.9

    max_r = -1000
    stats = []

    for i in range(N_EPISODES):
        eps = max(eps * de, min_eps)
        state = transform_state(env.reset())
        total_reward = 0
        steps = 0
        done = False
        reward_buffer = deque(maxlen=N_STEP)
        state_buffer = deque(maxlen=N_STEP)
        action_buffer = deque(maxlen=N_STEP)
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = transform_state(next_state)
            total_reward += reward
            reward += 10 * np.abs(next_state[1])
            steps += 1
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            if len(reward_buffer) == N_STEP:
                dqn.memory.push(state_buffer[0], action_buffer[0], next_state,
                                sum([(GAMMA ** i) * r for i, r in enumerate(reward_buffer)]))
                dqn.update()
            state = next_state

        if len(reward_buffer) == N_STEP:
            rb = list(reward_buffer)
            for k in range(1, N_STEP):
                dqn.memory.push(state_buffer[k], action_buffer[k], next_state,
                                sum([(GAMMA ** i) * r for i, r in enumerate(rb[k:])]))
                dqn.update()

        if i % TARGET_UPDATE == 0:
            dqn.target_model.load_state_dict(dqn.model.state_dict())
            if i % (5 * TARGET_UPDATE) == 0:
                dqn.save()

        stats.append(total_reward)
        print('Episodes:', i, 'Total reward:', total_reward)

        if total_reward > -110:
            avr = np.mean(stats[-20:])
            print('avr:', avr)
            if avr > -130:
                dqn.save(f"agent{int(avr)}.pkl")

    plt.plot(stats)
    plt.show()
