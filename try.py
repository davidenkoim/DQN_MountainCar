import gym
from .agent import Agent

ag = Agent('agent-120.pkl')
env = gym.make('MountainCar-v0')

def avr_reward(ag, env, n=20):
    """Counting average score"""
    avr_r = 0
    for _ in range(n):
        state = env.reset()
        reward = 0
        done = False
        while not done:
            action = ag.act(state)
            state, r, done, _ = env.step(action)
            reward += r
        avr_r += reward
    return avr_r / n

print(avr_reward(ag, env))

# You can see how car climbs the hill
state = env.reset()
reward = 0
done = False
while not done:
    env.render()
    action = ag.act(state)
    state, r, done, _ = env.step(action)
    reward += r
env.close()
print('Reward:', reward)
