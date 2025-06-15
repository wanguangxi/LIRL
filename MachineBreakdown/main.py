#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :main.py
@Description :
@Date        :2022/01/02 11:11:22
@Author      :wgx
@Version     :1.0
'''

from copy import copy
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import task_set as TS
import environment as EA
import matplotlib.pyplot as plt

#Hyperparameters
# lr_mu        = 0.0005
lr_mu        = 0.0005
lr_q         = 0.0001
gamma        = 0.8
batch_size   = 32
buffer_limit = 50000
tau          = 0.005 # for target network soft update


loss = []


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

class MuNet(nn.Module):
    def __init__(self,state_size, action_size):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))# [0,1]
        return mu

class QNet(nn.Module):
    def __init__(self,state_size, action_size):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_size, 64)
        self.fc_a = nn.Linear(action_size,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32,1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
      
def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s,a,r,s_prime,done_mask  = memory.sample(batch_size)
    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.smooth_l1_loss(q(s,a), target.detach())
    loss.append(q_loss.item())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s,mu(s)).mean() # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()
    
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def task_choose(n,mu):

    if n ==49:
        n = 48.5
    if n ==98:
        n = 97.5

    x = np.arange(0,1,1.00/n)
    # print(x)
    # print("mu:", mu)
    # print("x.shape[0]",x.shape[0])
    for i in range(x.shape[0]):
        if i == x.shape[0]-1:
            return x.shape[0]-1
        if mu >=x[i] and mu <x[i+1]:
            return i



class RL_Actor:
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.state_size = len(self.env.state) 
        # print("self.state_size:", self.state_size )
        self.pro_num = self.env.pro_num
        self.task_set = TS.TaskSet(self.pro_num)

        self.s = self.env.state
        self.action_size = self.env.PM_num*2
        self.Score_record = []

        self.memory = ReplayBuffer()
        # self.memory = memory
        self.q = QNet(self.state_size,self.action_size)
        self.q_target = QNet(self.state_size,self.action_size)
        self.q_target.load_state_dict(self.q.state_dict())

        self.mu = MuNet(self.state_size,self.action_size)
        self.mu_target = MuNet(self.state_size,self.action_size)
        self.mu_target.load_state_dict(self.mu.state_dict())
      

        self.score = 0.0
        self.print_interval = 20

        self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=lr_mu)
        self.q_optimizer  = optim.Adam(self.q.parameters(), lr=lr_q)
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(self.env.PM_num))


    def action_choose(self, a):

        
        mu = a.tolist()
        if abs(mu[0]) >1:
            mu[0] = 1
        if abs(mu[1]) >1:
            mu[1] = 1            

        for i in range (len(mu)):
            mu[i] = abs(mu[i])



        N_can = []
        action = [None, None, None]
        task_state = self.s.reshape(2,-1)
        Nt_c = task_state[0].reshape(-1,5)
        # Nt_a = task_state[1].reshape(-1,5)
        for i in range(Nt_c.shape[0]):
            for j in range(Nt_c.shape[1]):
                if Nt_c[i][j]== 0:
                    N_can.append(self.task_set.task_set[i][j])
                    break     

        if len(N_can) >= 1:
            item = task_choose(len(N_can),mu[0]) 
            action[0] = N_can[item]   
        else:
            action[0] = self.task_set.task_set[-1][-1]  
        task0 = action[0]
    
        task0.theta = task0.ref_time[0]+(task0.ref_time[1] - task0.ref_time[0])*abs(mu[1])

        action[0] = task0
        return action


        
    # def state_receive(self, message):

    #     print("****************")
    #     s_prime, r, done, info = message.get()  #转换成np.array
    #     print(s_prime, r, done, info)
    #     self.memory.put((copy.deepcopy(self.s),copy.deepcopy(self.a),r/100.0,s_prime,done))
    #     self.score +=r
    #     self.s  = s_prime
  
    def run(self, episodes):
        
        for episode in range(episodes):
            s0 = self.env.reset()
            self.s = np.array(s0)
            # print("S0:",s0)
            self.done = False

            while not self.done:
                self.a = self.mu(torch.from_numpy(self.s).float())  #选择动作
                noise = self.ou_noise()

                self.a  = self.a.detach().numpy()
                # print("self.a:" ,self.a)
                # if episode >= 3000 and episode <= 3500:
                #     self.a[noise.size-1:-1]  = self.a[noise.size-1:-1]+ noise
                #     # print("self.a + noise:" ,self.a)
                #
                #     # while True :
                #     #     pass
                # elif episode >= 4500:
                #     self.a[noise.size-1:-1]  = self.a[noise.size-1:-1]+ noise
                # else:
                #     self.a[noise.size-1:-1]  = self.a[noise.size-1:-1]

                self.a[noise.size - 1:-1] = self.a[noise.size - 1:-1] + noise
                action = self.action_choose(self.a)
                r1 = self.env.step(action)
                
                # print("r1:",r1)
                # print("r2:",r2)
                # print("r3:",r3)
            
                # print("s_prime:",s_prime)
                # print("reward:", r)
                # print("Done:", done)
                s_prime, r, done, info = r1

                # print("type s:", type(self.s))
                # print("type a:", type(self.a))
                # print("type s_prime:", type(s_prime))
                # print("type r:", r)


                self.memory.put((self.s,self.a,r,s_prime,done))
                self.score +=r
                # print("s_prime:",s_prime)
                self.s = s_prime
                self.done = done

            self.Score_record.append(self.score) 
            self.score = 0.0
            if self.memory.size()>2000:
                for i in range(10):
                    train(self.mu, self.mu_target, self.q, self.q_target, self.memory, self.q_optimizer, self.mu_optimizer)
                    soft_update(self.mu,self.mu_target)
                    soft_update(self.q,  self.q_target)
            
            # if episode%self.print_interval==0 and episode!=0:
            #     # print("# of episode :{}, avg score : {:.1f}".format(episode, self.score/self.print_interval))
            #     # self.Score_record.append(self.score/self.print_interval)  
            #     self.score = 0.0
                # self.Score_record.append(self.score/self.print_interval) 
 
        
        # x = range(len(self.Score_record))
        # plt.plot(x,self.Score_record)
        # plt.show()




if __name__ == '__main__':

    import time
    env = EA.Env(pro_num=5, alpha=0.2, beta=0.8)
    rl = RL_Actor(env)
    t1 = time.time()
    rl.run(2000)
    t2 = time.time()
    print("task 5*5 running time:",t2-t1)

    data_save = np.array(rl.Score_record)
    np.save("5_Score_record.npy",data_save)
    loss_data = np.array(loss)
    np.save("5_Loss_record.npy", loss_data)

    loss = []



    env = EA.Env(pro_num=10, alpha=0.2, beta=0.8)
    rl = RL_Actor(env)
    rl.run(2000)
    data_save = np.array(rl.Score_record)
    np.save("10_Score_record.npy",data_save)

    loss_data = np.array(loss)
    np.save("10_Loss_record.npy", loss_data)
    loss = []

    env = EA.Env(pro_num=15, alpha=0.2, beta=0.8)
    rl = RL_Actor(env)
    rl.run(2000)
    data_save = np.array(rl.Score_record)
    np.save("15_Score_record.npy",data_save)
    loss_data = np.array(loss)
    np.save("15_Loss_record.npy", loss_data)

    loss = []
    env = EA.Env(pro_num=25, alpha=0.2, beta=0.8)
    rl = RL_Actor(env)
    rl.run(2000)
    data_save = np.array(rl.Score_record)
    np.save("25_Score_record.npy",data_save)
    loss_data = np.array(loss)
    np.save("25_Loss_record.npy", loss_data)



    # env = EA.Env(pro_num=25, alpha=0.8, beta=0.2)
    # rl = RL_Actor(env)
    # rl.run(5000)
    # data_save = np.array(rl.Score_record)
    # np.save("25_Score_record.npy",data_save)

    # env = EA.Env(pro_num=35, alpha=0.8, beta=0.2)
    # rl = RL_Actor(env)
    # rl.run(5000)
    # data_save = np.array(rl.Score_record)
    # np.save("35_Score_record.npy",data_save)

    # env = EA.Env(pro_num=45, alpha=0.8, beta=0.2)
    # rl = RL_Actor(env)
    # rl.run(5000)
    # data_save = np.array(rl.Score_record)
    # np.save("45_Score_record.npy",data_save)


    # env = EA.Env(pro_num=55, alpha=0.8, beta=0.2)
    # rl = RL_Actor(env)
    # rl.run(5000)
    # data_save = np.array(rl.Score_record)
    # np.save("55_Score_record.npy",data_save)


    # x = range(len(loss))
    # plt.plot(x,loss)
    # plt.ylabel("Loss value")
    # plt.xlabel("Training epochs")
    # plt.show()
