import gym
from hw02_lunar_lander.agent import Agent

ag = Agent('agent-120.pkl')
env = gym.make('MountainCar-v0')
# env.seed(24)

def avr_reward(ag, env, n=20):
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
