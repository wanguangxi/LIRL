import collections
import numpy as np
import env as ENV
import ae as AE
import random
import torch
import torch.nn as nn
import pickle
buffer_limit = 100000

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)


def main(num_of_jobs, num_of_robots,alpha,beta,episodes):

    env = ENV.Env(num_of_jobs, num_of_robots,alpha,beta)
    
    memory = ReplayBuffer()
    action = np.array([0,0,0.00])
    state_size= len(env.state)
    action_size = len(env.action)
    #加载自动编码器
    hidden_dims = [16,64,8]
    autoencoder = AE.Autoencoder(action_size, hidden_dims)
    autoencoder.load_state_dict(torch.load("autoencoder.pth"))
    latent_action = hidden_dims[-1]
    job_list = np.arange(0,num_of_jobs,1)
    np.random.shuffle(job_list)
    for episode in range(episodes):
        s = env.reset()
        # print("*****************")
        for op in range(5):
        # for job in job_list:
            for job in job_list:
                
                action[0] = job
                robot_list = []
                if env.robot_state[0]==1:
                    robot_list.append(0)
                if env.robot_state[1]==1:
                    robot_list.append(1)
                if env.robot_state[2]==1:
                    robot_list.append(2)
                robot_list = np.array(robot_list)
                np.random.shuffle(robot_list)
                action[1] = robot_list[0]
                action[2] = random.uniform(0,1)
                s_prime, r, done= env.step(action)
                # print("action:",action)
                # print("reward:",r)
                # print("done:",done)
                action_ = torch.tensor(action)
                
                action_=action_.to(torch.float32)
                latent_action = autoencoder.encoder(action_)
                # print("latent_action:",latent_action)
                memory.put((s,latent_action.detach().numpy(),r,s_prime,done))
                s = s_prime
                # print("state:",s)

        # print("state:",s)
    with open("human.pkl", "wb") as file:
        pickle.dump(memory, file)
  
if __name__ == '__main__':
    num_of_jobs = 5
    num_of_robots = 3
    alpha = 0.2
    beta = 0.8
    episodes = 10000
    main(num_of_jobs, num_of_robots,alpha,beta,episodes)            