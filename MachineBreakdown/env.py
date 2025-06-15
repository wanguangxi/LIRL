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
import gant as gant
import matplotlib.pyplot as plt


repair_time = 10

# 完全柔性制造系统
class Env:
    def __init__(self, num_of_jobs, num_of_robots,breakdown_times, breakdown_rate,alpha,beta,render= False):
       
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
        self.breakdown_times = breakdown_times
        self.breakdown_cnt = 0
        self.breakdown_rate = breakdown_rate
        self.alpha = alpha
        self.beta = beta
        self.step_count = 0

        self.gant_macInfo = []#机器
        self.gant_flow = [] #实际加工时间
        self.gant_startTime = [] #开始加工时间
        self.gant_workpiece = [] #job
        self.gant_operation = [] #操作
        self.rendering = render
        self.render_cnt = 0


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
        self.render_cnt = 0
        self.breakdown_cnt = 0
        self.gant_macInfo = []#机器
        self.gant_flow = [] #实际加工时间
        self.gant_startTime = [] #开始加工时间
        self.gant_workpiece = [] #job
        self.gant_operation = [] #操作

        return self.state


    def step(self,action):

        for i in range (len(self.task_state)):
            if self.task_state[i] == 1:
                self.done = True
            else:
                self.done = False
                break

        #确定选择的操作
        job_id = round(action[0])  #取job id
        if job_id >=self.num_of_jobs:
            job_id = self.num_of_jobs -1
        self.gant_workpiece.append(job_id+1)


        operations = self.task_set[job_id]   
        assigned_num = 0              
        for iter in range(len(operations)):
            task = operations[iter]
            if task.state:
                assigned_num = assigned_num+1
            else:
                break                  
        self.gant_operation.append(assigned_num+1)        

        robot_id = round(action[1])      # int
        if robot_id>=len(self.robot_state)-1:
            robot_id = len(self.robot_state)-1

        self.gant_macInfo.append(robot_id+1)



        param = action[2]         # float[0,1]
        

        task.processing_robot = robot_id
        ##动作合法计算reward
        self.last_action_state[0] = 1
        self.last_action_state[1] = 1
 
        #计算加工时间
        C_duration = task.ref_time[1]
        E_duration = task.ref_energy[1]
        task.processing_time = task.ref_time[0]+C_duration*param # 任务处理时间，待增加干扰项 
        self.gant_flow.append(task.processing_time)
        #对时间线的改变作为时间的reward
        if assigned_num == 0: #首道工序
            task.start_time = max(self.robot_timeline[robot_id],self.current_time)                       #任务开始时间  
        else: #非首道工序
            pre_task = operations[assigned_num-1]
            task.start_time = max(self.current_time,pre_task.end_time,self.robot_timeline[robot_id]) 
        self.gant_startTime.append(task.start_time)
        
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

        #空闲机器
        idle_robot = []
        busy_robot = []
        for robot_id in range(len(self.robot_state)):
            if self.robot_state[robot_id]==1:
                 idle_robot.append(robot_id)
            else:
                busy_robot.append(robot_id)
        # print("idle robots:",idle_robot)
        # print("busy_robots:",busy_robot)
        #空闲机器故障
        if random.random()<self.breakdown_rate and self.breakdown_cnt<self.breakdown_times and idle_robot:
            self.breakdown_cnt+=1
            breakdown_robot_id = random.choice(idle_robot) 

            # print("breakdown robot id:", breakdown_robot_id+1)
            #更新gant图
            self.gant_macInfo.append(breakdown_robot_id+1)
            self.gant_flow.append(repair_time)
            self.gant_workpiece.append(0)
            self.gant_operation.append(0)
            self.gant_startTime.append(self.robot_timeline[breakdown_robot_id])
            #更新时间线
            self.robot_timeline[breakdown_robot_id] = self.robot_timeline[breakdown_robot_id] +repair_time
            self.robot_state[breakdown_robot_id] = 0
            #当前时间线更新：= 最先空闲的机器人的时间线
            self.current_time = np.min(self.robot_timeline)
            for robot in range(len(self.robot_state)):
                if self.robot_timeline[robot]<= self.current_time:
                    self.robot_state[robot] =1
       
        #正在加工的机器故障
        if random.random()<self.breakdown_rate and self.breakdown_cnt<self.breakdown_times and busy_robot:
            self.breakdown_cnt+=1
            breakdown_robot_id = random.choice(busy_robot) 
            # print("breakdown robot id:", breakdown_robot_id+1)
            #找到故障机器加工的作业编号
            indices = [i for i in range(len(self.gant_macInfo)) if self.gant_macInfo[i] == breakdown_robot_id+1]
            index = indices[-1]
            job_processing = self.gant_workpiece[index]-1
            operation_processing = self.gant_operation[index]-1
            flow_time = self.gant_flow[index]
            self.gant_flow[index] = self.current_time - self.gant_startTime[index]
            #更行甘特图
            self.gant_macInfo.append(breakdown_robot_id+1)
            self.gant_flow.append(repair_time)
            self.gant_workpiece.append(0)
            self.gant_operation.append(0)
            self.gant_startTime.append(self.current_time)

            #更新时间线
            self.robot_timeline[breakdown_robot_id] = self.robot_timeline[breakdown_robot_id]-(self.gant_startTime[index]+flow_time-self.current_time)+repair_time
            self.robot_state[breakdown_robot_id] = 0
            self.task_set[job_processing][operation_processing].state = False

            #当前时间线更新：= 最先空闲的机器人的时间线
            self.current_time = np.min(self.robot_timeline)
            for robot in range(len(self.robot_state)):
                if self.robot_timeline[robot]<= self.current_time:
                    self.robot_state[robot] =1   
            self.task_state[job_processing*5+operation_processing] = 0


        self.state = np.concatenate((self.task_state,self.robot_state))
        self.state = np.concatenate((self.state,self.task_prcoessing_time_state))
        self.state = np.concatenate((self.state,self.last_action_state))

        for i in range (len(self.task_state)):
            if self.task_state[i] == 1:
                self.done = True
            else:
                self.done = False
                break       

        if self.done:
            self.reward = len(self.task_state)
            if self.rendering:
                plt.cla()
                gant.gantt(self.gant_macInfo,self.gant_flow,self.gant_startTime,self.gant_workpiece,self.gant_operation)
                plt.pause(1)
            
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