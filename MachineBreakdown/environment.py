#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :environment.py
@Description :
@Date        :2022/01/02 20:27:01
@Author      :wgx
@Version     :1.0
'''
import random
import time

# from gym.core import RewardWrapper
import energy_model as EM
import task_set as TS
from math import pi
import numpy as np

global s_prime
PM_dict={
    'PMx':0,
    'PM0':0,
    'PM1':1,
    'PM2':2,
}
   

class PMActor:
    def __init__(self, obsever, name = 'PMx'):
        super().__init__()
        self.name = name
        self.obsever = obsever
        self.state = False    # False - idle; True - Busy
        self.id = PM_dict[self.name]
    def reset(self):
        self.state = False
    def process(self, task):
        # calculate value
        # print("Processing Task id:", task.id)
        self.state = True
        self.obsever.update_PM_state(self.id, self.state)
        self.obsever.update_Bt(task)
        # time.sleep(task.theta)
        # print("receive task", task.type)
        energy = EM.energy_dynamic(task.target_position,task.mass,task.theta)
        # print("energy consuming", energy)
        task.set_energy_consuming(energy)
        self.obsever.update_St(task)
        self.obsever.update_reward(task)
        self.state = False
        self.obsever.update_PM_state(self.id, self.state)
        # observation,reward, done, PM_state = self.obsever.cal(task)
        # print("PM",self.id)
        result = self.obsever.cal(task)
        return result


class ObserverActor:
    def __init__(self,product_num, alpha, beta):
        super().__init__()
        self.time_cnt = 0
        self.init_flag = True
        self.product_num = product_num
        self.Nt_c = [0 for i in range(self.product_num*5)]
        self.Nt_a = [0 for i in range(self.product_num*5)]
        self.Pt = self.create_Pt()
        self.Bt = [0 for i in range(self.product_num*5)]
        self.PM_state = [0,0,0]
        self.reward = 0
        self.done = False
        self.alpha = alpha
        self.beta = beta
        self.observation = None

    def create_Pt(self):
        Pt = [[] for i in range(self.product_num*5)]
        cnt = 0
        for i in range(self.product_num):
            Pt[cnt]   = (7.2+4.0)/2
            Pt[cnt+1] = (14.20+2.0)/2
            Pt[cnt+2] = (16.5+2.5)/2
            Pt[cnt+3] = (18.0+2.1)/2
            Pt[cnt+4] = (16.8+2.4)/2
            cnt = cnt+5
        return Pt

    def reset(self):
        self.time_cnt = 0
        self.init_flag = True
        self.Nt_c = [0 for i in range(self.product_num*5)]
        self.Nt_a = [0 for i in range(self.product_num*5)]
        self.Pt = self.create_Pt()
        self.Bt = [0 for i in range(self.product_num*5)]
        self.PM_state = [0]
        self.reward = 0
        self.done = False
        self.observation = np.array(self.Nt_c+self.Pt)
        return self.observation 
        



    def update_Bt(self,task):
        self.Bt[task.id] = self.time_cnt
    def update_PM_state(self,PM, state):
        self.PM_state[PM] = state

    def update_St(self, task):
        self.Nt_c[task.id] = 1
        self.Nt_a[task.id] = 1
        self.Pt[task.id] = task.theta
        
        
        for i in range (len(self.Nt_c)):
            if self.Nt_c[i] == 1:
                self.done = True
            else:
                self.done = False
                break



        # if task.id == (self.product_num*5-1):
        #     self.done = True
        # else:
        #     self.done = False

    def update_reward(self,task):
        C_duration = task.ref_time[1] - task.ref_time[0]
        E_duration = task.ref_energy[1] - task.ref_energy[0]

        # self.reward = -1*(self.alpha*((task.ref_time[1] - task.theta)/C_duration) +\
        #                self.beta*((task.ref_energy[1] - task.energy_consuming)/E_duration))



        self.reward = -1*(self.alpha*((task.theta-task.ref_time[0])/(C_duration*(1+ random.random()))) +\
                       self.beta*((task.energy_consuming -task.ref_energy[0] )/E_duration))


        # self.reward = -1*(self.alpha+self.beta*((task.ref_energy[1] - task.energy_consuming)/E_duration))

        # print("task.ref_time[1] - task.theta", task.ref_time[1] - task.theta)    
        # print("task.ref_energy[1] - task.energy_consuming",task.ref_energy[1] - task.energy_consuming)
        # print("reward:", self.reward)
        if self.reward>0:
            print((task.ref_energy[1] - task.energy_consuming))

            print("reward >0")
            quit()

      
    def cal(self, task):
        if self.init_flag:
            self.time_cnt = 0
            self.init_flag = False
        else:
            self.time_cnt = self.time_cnt+task.theta

        # print("times of obsevation:", self.time_cnt)
        # print("observation:", self.Nt_c, self.Nt_a, self.Pt, self.Bt,self.reward,self.done,self.PM_state)
        observation = np.array(self.Nt_c+self.Pt)
        # print("observation:", self.Nt_c, self.Nt_a, self.Pt, self.Bt)
        # print("observation1:",observation)
        
        return observation,self.reward, self.done, self.PM_state


class Env:
    def __init__(self,pro_num, alpha, beta):
        super().__init__()
        self.pro_num = pro_num
        self.PM_num = 1
        self.observer = ObserverActor(pro_num,alpha,beta)
        self.pm0 = PMActor(self.observer,"PM0")
        # self.pm1 = PMActor(self.observer,"PM1")
        # self.pm2 = PMActor(self.observer,"PM2")
        self.state = self.reset()

    def reset(self):
        self.observer.reset()
        self.pm0.reset()
        # self.pm1.reset()
        # self.pm2.reset()
        # print("reset success!!!")
        return self.observer.observation

    def step(self, action):
        # self.action_dispatcher.tell(action)
        result1 = self.pm0.process(action[0])
        # result2 = self.pm1.process(action[1])
        # result3 = self.pm2.process(action[2])
        return result1



