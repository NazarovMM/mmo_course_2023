
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
import time
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from random import random, sample
from collections import deque
from utils import FrameStackingAndResizingEnv
from dqn_model import DQN
from torch.optim import Adam
import matplotlib.pyplot as plt


# Определите класс буфера воспроизведения для хранения и выборки переходов
class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.idx = 0

    def insert(self, sars):
        self.buffer.append(sars)

    def sample(self, num_samples):
        if num_samples > len(self.buffer):
            return sample(self.buffer, len(self.buffer))
        return sample(self.buffer, num_samples)

learning_rate = 0.0001

# Определите класс памяти для представления переходов состояний
class memory:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

# Функция для обновления целевой модели путем копирования весов из основной модели
def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())

# Функция для одного шага обучения
def train_step(model, state_transitions, tgt, num_actions, device, gamma=0.99):
    # Преобразовать переходы состояний в тензоры и перенести их на CPU
    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions])).to(device)
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions])).to(device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).to(device)
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions])).to(device)
    actions = [s.action for s in state_transitions]
    
    # Вычислить максимальное значение Q для следующих состояний, используя целевую модель
    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0]  
    
    # Сброс градиентов оптимизатора модели
    model.optimizer.zero_grad()
    # Вычислить Q-значения для текущих состояний
    qvals = model(cur_states) 
    # Преобразование списока действий в тензор с однократным кодированием
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)

    # Вычисление прогнозируемого значения Q
    qvals_pred = torch.sum(qvals * one_hot_actions, -1)

    # Вычисление целевого значения Q
    qvals_target = rewards.squeeze() + mask[:, 0] * qvals_next * gamma

    # Рассчет loss, используя среднюю квадратичную ошибку
    loss = ((qvals_pred - qvals_target) ** 2).mean()

    # Обратное распространение потерь и обновление весов модели
    loss.backward()
    model.optimizer.step()
    return loss


# Функция для обучения модели DQN
def train(file, name='breakout',  device="cpu",test=False):
    # Определение параметров обучения
    min_rb_size = 50000
    sample_size = 32
    lr = 0.0001
    num_iterations = 15_000_000
    eps_decay = 0.999999

    env = gym.make("Breakout-v0",)
    env = FrameStackingAndResizingEnv(env, 84, 84, 4)
    last_observation = env.reset()

    # Создание основной модель DQN и загрузка сохраненной модель, если мы находимся в тесте
    model = DQN(env.observation_space.shape, env.action_space.n).to(device)
    if test:
        model.load_state_dict(torch.load(file))
    # Создание целевой модель DQN и её обновления, чтобы она соответствовала основной модели
    target = DQN(env.observation_space.shape, env.action_space.n).to(device)
    update_tgt_model(model, target)
    
    # Создание ReplayBuffer
    rb = ReplayBuffer()
    steps_since_train = 0
    loss_set = []

    step_num = -1 * min_rb_size
    episode_rewards = []
    rolling_reward = 0
    mean_reward_per_episode = 0
    # Запуск цикла обучения
    for i in range(num_iterations):
        observation = env.reset()
        done = False
        rolling_reward = 0
        while not done:
            # Вычислить коэффициент epsilon
            eps = eps_decay ** (step_num)

            # Выбор действия с помощью эпсилон-жадной разведки
            if random() < eps:
                action = (env.action_space.sample()) 
            else:
                action = model(torch.Tensor(last_observation).unsqueeze(0).to(device)).max(-1)[-1].item() 
            
            # Шаг в среде
            observation, reward, done, info = env.step(action)
            rolling_reward += reward

            # Сохранение перехода состояний в буфер
            rb.insert(memory(last_observation, action, reward, observation, done))

            last_observation = observation

            if done:
                # Добавление вознаграждения в список вознаграждений эпизода
                episode_rewards.append(rolling_reward)
                # print("average_reward : ",episode_rewards)
                
                observation = env.reset()
                loss = train_step(model, rb.sample(sample_size), target, env.action_space.n, device).detach().cpu().numpy()
                loss_set.append(loss.tolist())

        print("number of iters :", i)
        print("rewards (unclipped): ",rolling_reward)
       
    env.close()


    fig, axs = plt.subplots(2)
    fig.suptitle('loss vs number of episodes and rewards per episodes plot')
    axs[0].plot(range(1,len(loss_set)+1),loss_set)
    axs[1].plot(range(1,len(episode_rewards)+1),episode_rewards)
    plt.show()


# Функция для тестирования модели DQN
def test(file, name='breakout', device="cpu", test=True):
    num_iterations = 100
    initiate_sequence = 5

    env = gym.make("Breakout-v0", render_mode="human")
    env = FrameStackingAndResizingEnv(env, 84, 84, 4)

    model = DQN(env.observation_space.shape, env.action_space.n).to(device)
    if file:
        model.load_state_dict(torch.load(file, map_location=device))
    target = DQN(env.observation_space.shape, env.action_space.n).to(device)
    update_tgt_model(model, target)

    rb = ReplayBuffer()
    episode_rewards = []

    # Запуск цикла тестирования
    for _ in range(num_iterations):
        observation = env.reset()
        last_observation = observation
        done = False
        rolling_reward = 0
        while not done:
            env.render(mode='rgb_array')
            eps = 0

            if random() < eps or (test and initiate_sequence > 0):
                action = env.action_space.sample()
                initiate_sequence -= 1
            else:
                action = model(torch.Tensor(last_observation).unsqueeze(0).to(device)).max(-1)[-1].item() 

            observation, reward, done, info = env.step(action)
            rolling_reward += reward
            rb.insert(memory(last_observation, action, reward, observation, done))
            last_observation = observation

            if done:
                episode_rewards.append(rolling_reward)
                print("reward per episode, unclipped :", rolling_reward)
    env.close()
    print("Mean reward for 100 episodes: ", np.mean(episode_rewards))



if __name__ == "__main__":
  
    # train(file=None)
    device = torch.device('cpu')
    test(file = 'D:/MyCode/mmo_course/Lab6.1/trained_DQN.pth')

