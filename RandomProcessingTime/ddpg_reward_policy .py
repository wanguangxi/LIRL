import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import env_reward as ENV
import pickle
import math
# random.seed(20230826)
#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.98
batch_size   = 256
buffer_limit = 100000000
tau          = 0.005 # for target network soft update

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
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        # print(s_lst.size())    
        s_lst_= torch.FloatTensor(s_lst)
        a_lst_= torch.tensor(a_lst, dtype=torch.float)
        r_lst_= torch.tensor(r_lst, dtype=torch.float)
        s_prime_lst_ = torch.tensor(s_prime_lst, dtype=torch.float)
        done_mask_lst_ = torch.tensor(done_mask_lst, dtype=torch.float)

        return s_lst_,a_lst_,r_lst_,s_prime_lst_,done_mask_lst_
    
    def size(self):
        return len(self.buffer)
    
outlayer = nn.Sigmoid()
class MuNet(nn.Module):
    def __init__(self,state_size, action_size):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = outlayer(self.fc_mu(x)) 
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
        self.theta, self.dt, self.sigma = 0.1, 0.05, 0.2
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def action_choose(env,a):
    a_ = a.detach().numpy()
    job_i = a_[0]
    robot_i = a_[1]
    
    valid_job =[]
    valid_robot = []
    task_set = env.task_set
    # for job_id in range(len(task_set)):
    #     job = task_set[job_id]
    #     finished = False
    #     for op in range(len(task_set[job_id])):
    #         task = job[op]
    #         if task.state:
    #             finished = True
    #         else:
    #             finished = False
    #             break
    #     if not finished:
    #         valid_job.append(job_id)

    # for robot_id in range(len(env.robot_state)):
    #     if env.robot_state[robot_id]==1:
    #         valid_robot.append(robot_id)


    for job_id in range(len(task_set)):
    
        valid_job.append(job_id)         
   
    
    for robot_id in range(len(env.robot_state)):
       
        valid_robot.append(robot_id)


    # 选择规则[0,1]划分为n等份，观察选择因子所在的位置
    if len(valid_job)>0:
        delta = 1.00/len(valid_job)
        r = math.floor(job_i/delta)
        if job_i ==1:
            job_i = valid_job[-1]
        else:
            job_i = valid_job[r]
    if len(valid_robot)>0:
        delta = 1.00/len(valid_robot)
        r = math.floor(robot_i/delta)
        if robot_i ==1:
            robot_i =valid_robot[-1]
        else:
            robot_i =valid_robot[r] 
    

    return [job_i, robot_i, a_[2]]







def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s,a,r,s_prime,done_mask  = memory.sample(batch_size)
    

    target =torch.unsqueeze(r, dim=1)+ gamma * q_target(s_prime, mu_target(s_prime)).mul(done_mask) 
    q_loss = F.smooth_l1_loss(q(s,a), target.detach())
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
    
def main(num_of_jobs, num_of_robots,alpha,beta,episodes):

    env = ENV.Env(num_of_jobs, num_of_robots,alpha,beta)
    state_size= len(env.state)
    action_size = len(env.action)
    #加载自动编码器
    score_record = []
    
    q, q_target = QNet(state_size,action_size), QNet(state_size,action_size)
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(state_size,action_size), MuNet(state_size,action_size)
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 20

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer  = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_size))
    memory = ReplayBuffer()


    for n_epi in range(episodes):
        s = env.reset()
        # print(s)
        done = False
        # print("episode:",n_epi)
        steps = 0
        print("episode= ",n_epi)
        
        while not done:
            a = mu(torch.from_numpy(s).float()) 
            # print("latent_action:",a)
            
            a = a + torch.from_numpy(ou_noise())   
            
            a = torch.clamp(a, 0, 1)
            a=a.to(torch.float32)
            action = action_choose(env,a)

            # print("action:",action)
            s_prime, r, done= env.step(action)
            # print("reward:",r)
            # print("done:",done)
            memory.put((s,a.detach().numpy(),r,s_prime,done))
            score += r
            s = s_prime
            # print("state:",s)
            steps += 1
        print(steps)    
        score_record.append(score)
        score = 0.0  
        if memory.size()>500:
            for i in range(20):
                # print("Training time:",i)
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q,  q_target)
        # if n_epi%print_interval==0 and n_epi!=0:
        #     print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
        #     score_record.append(score/print_interval)
        #     score = 0.0             

    
    torch.save(mu.state_dict(), "mu_model.pth")
    torch.save(mu_target.state_dict(), "mu_target_model.pth")
    torch.save(q.state_dict(), "q_model.pth")
    torch.save(q_target.state_dict(), "q_target_model.pth")
    return score_record


if __name__ == '__main__':
    print("5 0.2 0.8")
    num_of_jobs = 5
    num_of_robots = 3
    alpha = 0.2
    beta = 0.8
    episodes = 2000
    score_records = []
    for i in range(50):
        score_records.append(main(num_of_jobs, num_of_robots,alpha,beta,episodes))
    score_records=np.array(score_records)
    np.save('job5_score_records_0.2_0.8_co_reward.npy',score_records)
    print("training done!")
    import matplotlib.pyplot as plt
    x = range(len(score_records[0]))
    y =  score_records[0]
    plt.plot(x,y)
    plt.show()
    # print("5 0.8 0.2")
    # num_of_jobs = 5
    # num_of_robots = 3
    # alpha = 0.8
    # beta = 0.2
    # episodes = 2000
    # score_records = []
    # for i in range(50):
    #     score_records.append(main(num_of_jobs, num_of_robots,alpha,beta,episodes))
    # score_records=np.array(score_records)
    # np.save('job5_score_records_0.8_0.2_co.npy',score_records)

    # print("10 0.2 0.8")
    # num_of_jobs = 10
    # num_of_robots = 3
    # alpha = 0.2
    # beta = 0.8
    # episodes = 2000
    # score_records = []
    # for i in range(50):
    #     score_records.append(main(num_of_jobs, num_of_robots,alpha,beta,episodes))
    # score_records=np.array(score_records)
    # np.save('job10_score_records_0.2_0.8_energy.npy',score_records)

    # print("10 0.8 0.2")
    # num_of_jobs = 10
    # num_of_robots = 3
    # alpha = 0.8
    # beta = 0.2
    # episodes = 2000
    # score_records = []
    # for i in range(50):
    #     score_records.append(main(num_of_jobs, num_of_robots,alpha,beta,episodes))
    # score_records=np.array(score_records)
    # np.save('job10_score_records_0.8_0.2_energy.npy',score_records)

    # print("20 0.2 0.8")
    # num_of_jobs = 20
    # num_of_robots = 3
    # alpha = 0.2
    # beta = 0.8
    # episodes = 2000
    # score_records = []
    # for i in range(50):
    #     score_records.append(main(num_of_jobs, num_of_robots,alpha,beta,episodes))
    # score_records=np.array(score_records)
    # np.save('job20_score_records_0.2_0.8_energy.npy',score_records)

    # print("20 0.8 0.2")
    # num_of_jobs = 20
    # num_of_robots = 3
    # alpha = 0.8
    # beta = 0.2
    # episodes = 2000
    # score_records = []
    # for i in range(50):
    #     score_records.append(main(num_of_jobs, num_of_robots,alpha,beta,episodes))
    # score_records=np.array(score_records)
    # np.save('job20_score_records_0.8_0.2_energy.npy',score_records)

    # print("50 0.2 0.8")
    # num_of_jobs = 50
    # num_of_robots = 3
    # alpha = 0.2
    # beta = 0.8
    # episodes = 2000
    # score_records = []
    # for i in range(50):
    #     score_records.append(main(num_of_jobs, num_of_robots,alpha,beta,episodes))
    # score_records=np.array(score_records)
    # np.save('job50_score_records_0.2_0.8_energy.npy',score_records)

    # print("50 0.8 0.2")
    # num_of_jobs = 50
    # num_of_robots = 3
    # alpha = 0.8
    # beta = 0.2
    # episodes = 2000
    # score_records = []
    # for i in range(50):
    #     score_records.append(main(num_of_jobs, num_of_robots,alpha,beta,episodes))
    # score_records=np.array(score_records)
    # np.save('job50_score_records_0.8_0.2_energy.npy',score_records)
