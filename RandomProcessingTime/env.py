#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :environment.py
@Description :
@Date        :2023/08/23 20:27:01
@Author      :wgx
@Version     :1.0
'''
import random
import time
import energy_model as EM
import task_set as TS
from math import pi
import numpy as np



# 完全柔性制造系统
class Env:
    def __init__(self, num_of_jobs, num_of_robots,alpha,beta):
       
        self.num_of_jobs = num_of_jobs
        self.num_of_robots = num_of_robots
        self.task_set = TS.TaskSet(num_of_jobs).task_set
        self.task_state = np.zeros((5*num_of_jobs,), dtype = int) #0-未完成/分配 1-已分配
        self.task_prcoessing_time_state = np.zeros((5*num_of_jobs,), dtype = float) #0-未完成/分配 1-已分配

        self.robot_state = np.ones((num_of_robots,), dtype = int)   #0-不可用 1-可用
        self.last_action_state = np.ones((2,), dtype = int)# 11表示正常
        self.state = np.concatenate((self.task_state,self.robot_state)) # 所有operation的状态
        self.state = np.concatenate((self.state,self.task_prcoessing_time_state))
        self.state = np.concatenate((self.state,self.last_action_state))
        self.action = np.array([0,0,0.0]) #三个维度：operation, robot, and Parameter
        self.robot_timeline = np.zeros((num_of_robots,), dtype = float)
        self.current_time = 0.0
        self.future_time = 0.0
        self.done = False
        self.reward = 0.0
       
        self.alpha = alpha
        self.beta = beta

        self.step_count = 0

    def reset(self):
        self.task_set = TS.TaskSet(self.num_of_jobs).task_set
        self.task_state = np.zeros((5*self.num_of_jobs,), dtype = int) #0-未完成/分配 1-已分配
        self.robot_state = np.ones((self.num_of_robots,), dtype = int)   #0-不可用 1-可用
        self.task_prcoessing_time_state = np.zeros((5*self.num_of_jobs,), dtype = float) #0-未完成/分配 1-已分配
        self.last_action_state = np.ones((2,), dtype = int)# 11表示正常
        self.state = np.concatenate((self.task_state,self.robot_state)) # 所有operation的状态
        self.state = np.concatenate((self.state,self.task_prcoessing_time_state))        
        self.state = np.concatenate((self.state,self.last_action_state))

        self.action = np.array([0,0,0.0]) #三个维度：job, robot, and Parameter
        self.done = False
        self.robot_timeline = np.zeros((self.num_of_robots,), dtype = float)
        self.current_time = 0.0
        self.future_time = 0.0
        self.reward = 0.0
        self.step_count = 0
        return self.state

    def step(self,action):

        for i in range (len(self.task_state)):
            if self.task_state[i] == 1:
                self.done = True
            else:
                self.done = False
                break
        if self.done:
            self.reward = len(self.task_state)
            return self.state,self.reward,self.done  
      
        #确定选择的操作
        job_id = round(action[0])  #取job id
        if job_id >=self.num_of_jobs:
            job_id = self.num_of_jobs -1
        
        operations = self.task_set[job_id]   
        assigned_num = 0              
        for iter in range(len(operations)):
            task = operations[iter]
            if task.state:
                assigned_num = assigned_num+1
            else:
                break                  
                
        robot_id = round(action[1])      # int
        if robot_id>=len(self.robot_state)-1:
            robot_id = len(self.robot_state)-1
        param = action[2]         # float[0,1]
        # print("param:",param)

        ## 判断动作是否合法
        #1-选择了已经完成的job；2-工序选择正确，但是机器选择错误，机器不可用-10; 3-选择的任务，机器不支持加工
        if assigned_num == len(operations) and self.robot_state[robot_id] == 0:                    #1-选择了已经完成的job
            self.reward = -5
            self.state[-2] = 0
            self.state[-1] = 0
            return self.state,self.reward,self.done
        
        if assigned_num == len(operations) and self.robot_state[robot_id] == 1:                    #1-选择了已经完成的job
            self.reward = -5
            self.state[-2] = 0
            self.state[-1] = 1
            return self.state,self.reward,self.done        

        if self.robot_state[robot_id] == 0:                    #2-工序选择正确，但是机器选择错误，机器不可用-10
            self.reward = -5
            self.state[-1]  = 0
            self.state[-2]  = 1
            return self.state,self.reward,self.done  
        
        if not robot_id in task.available_modules:
            self.reward = -5
            self.state[-2] = 0
            self.state[-1] = 0
            return self.state,self.reward,self.done            
        task.processing_robot = robot_id
        ##动作合法计算reward
        self.last_action_state[0] = 1
        self.last_action_state[1] = 1
 
        #计算加工时间
        C_duration = task.ref_time[1]
        E_duration = task.ref_energy[1]
        task.processing_time = task.ref_time[0]+C_duration*param # 任务处理时间，待增加干扰项 
        #对时间线的改变作为时间的reward
        if assigned_num == 0: #首道工序
            task.start_time = max(self.robot_timeline[robot_id],self.current_time)                       #任务开始时间  
        else: #非首道工序
            pre_task = operations[assigned_num-1]
            task.start_time = max(self.current_time,pre_task.end_time,self.robot_timeline[robot_id]) 

        task.state = True   

        #机器人的时间线更新
        self.robot_timeline[robot_id] = task.start_time+task.processing_time
        task.end_time = task.start_time+task.processing_time
        #未来最大完工时间更新
        future_time_new = np.max(self.robot_timeline)
        delta_time = future_time_new-self.future_time
        self.future_time = future_time_new
        #当前时间线更新：= 最先空闲的机器人的时间线
        self.current_time = np.min(self.robot_timeline)

        #计算新增加的能耗
        delta_energy = EM.energy_dynamic(task.target_position,task.mass,task.processing_time)
        # print("delta_time:",delta_time)
        # print("C_duration:",C_duration)
        # print("delta_energy:",delta_energy)
        # print("E_duration:",E_duration)
        #计算reward
        self.reward = self.alpha*delta_time/C_duration + self.beta*delta_energy/E_duration
        self.reward = -1*self.reward


        #状态更新
        self.task_state[job_id*5+assigned_num] = 1
        self.task_prcoessing_time_state[job_id*5+assigned_num] = task.processing_time
        self.robot_state[robot_id] = 0
        for robot in range(len(self.robot_state)):
            if self.robot_timeline[robot]<= self.current_time:
                self.robot_state[robot] =1
        self.state = np.concatenate((self.task_state,self.robot_state))
        self.state = np.concatenate((self.state,self.task_prcoessing_time_state))
        self.state = np.concatenate((self.state,self.last_action_state))

        for i in range (len(self.task_state)):
            if self.task_state[i] == 1:
                self.done = True
            else:
                self.done = False
                break       
        return self.state, self.reward,self.done


if __name__ == '__main__': 

    num_of_jobs = 5
    num_of_robots = 3
    alpha = 0.2
    beta = 0.8
    env = Env(num_of_jobs, num_of_robots,alpha,beta)
    Operation_action = 1

    #随机策略测试
    import random
    from random import randint
    state = env.reset()
    for epsode in range(10):
        # Operation_action = randint(0,5*num_of_jobs-1)
         
        Robot_action = randint(0,num_of_robots-1)
        Param_action = random.uniform(0,1)
        action = np.array([Operation_action,Robot_action,Param_action])

        
        print("action:",action)


        state_, reward, done = env.step(action)
        print("*"*20)
        # print("epsode:",epsode)
        print("state:",state_)
        print("reward:",reward)
        print("done:",done)
        
        # if (state_== state).all():
        #     pass
        # else:
        #     Operation_action = Operation_action

        #     state = state_