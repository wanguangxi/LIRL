#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :main.py
@Description :
@Date        :2023/08/24 11:11:22
@Author      :wgx
@Version     :1.0
'''

import env as ENV
import numpy as np


num_of_jobs = 5
num_of_robots = 3
alpha = 0.2
beta = 0.8
episodes = 100
env = ENV.Env(num_of_jobs, num_of_robots,alpha,beta)
#随机策略测试
import random
from random import randint
gain_list = []
for episode in range(episodes):
    gain = 0
    state = env.reset()
    done = False
    while not done:
        Operation_action = randint(0,5*num_of_jobs-1)
        Robot_action = randint(0,num_of_robots-1)
        Param_action = random.uniform(0,1)
        action = np.array([Operation_action,Robot_action,Param_action])
        state_, reward, done = env.step(action)
        gain = gain + reward

    gain_list.append(gain)

import matplotlib.pyplot as plt
x = range(len(gain_list))
y =  gain_list
plt.plot(x,y)
plt.show()
 