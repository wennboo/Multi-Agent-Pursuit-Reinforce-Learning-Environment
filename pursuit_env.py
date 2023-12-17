import argparse
import math

from Agent import Agent
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def relative_angle(start_point,end_point):
    "end_point   start_point "
    x_diff = end_point.state.x - start_point.state.x
    y_diff = end_point.state.y - start_point.state.y
    relative_theta = math.atan(y_diff / (x_diff + 0.0001))

    if y_diff > 0 and x_diff < 0:
        relative_theta += math.pi
    elif y_diff < 0 and x_diff < 0:
        relative_theta -= math.pi
    return relative_theta


class Env:
    def __init__(self,args):
        self.num_pursuit = args.num_pursuit  #追捕者数量
        self.num_evasion = args.num_evasion  #逃跑者数量
        self.num_obscale = args.num_obstacle  #障碍物数量
        self.args = args  #超参数
        self.obs_shape_n = [(self.args.input_state_dim,)]*self.num_pursuit+[(self.args.input_state_dim,)]*self.num_evasion  #网络的输入维度
        self.action_space = self.action_space = [spaces.Discrete(self.args.action_dim)] * (self.num_pursuit+self.num_evasion) #智能体动作空间
        self.AgentsContains = {}  #智能体对象容器

    "创建智能体对象"
    def Init(self):
        "创建追捕者逃跑者障碍物对象"
        for i in range(self.num_pursuit + self.num_evasion + self.num_obscale):
            if i < self.num_pursuit:
                self.AgentsContains["pursuit_" + str(i + 1)] = Agent("pursuit_" + str(i + 1), self.obs_shape_n,
                                                                     self.action_space, i,self.args)
            elif i >= self.num_pursuit and i < self.num_pursuit + self.num_evasion:
                self.AgentsContains["evasion_" + str(i + 1 - self.num_pursuit)] = Agent(
                    "evasion_" + str(i + 1 - self.num_pursuit), self.obs_shape_n,
                    self.action_space, i,self.args)
            else:
                self.AgentsContains["obscale_" + str(i + 1 - self.num_pursuit - self.num_evasion)] = Agent(
                    "obscale" + str(i + 1 - self.num_pursuit - self.num_evasion), self.obs_shape_n,
                    self.action_space, i,self.args)

    "初始化环境,初始化智能体位置"
    def reset(self):
        if len(self.AgentsContains) == 0:
            self.Init()
        for agent in self.AgentsContains:
            random_x = np.random.uniform(0,self.args.map_width)
            random_y = np.random.uniform(0,self.args.map_height)
            random_theta = np.random.uniform(-math.pi,math.pi)
            self.AgentsContains[agent].state.x = random_x
            self.AgentsContains[agent].state.y = random_y
            self.AgentsContains[agent].state.theta = random_theta
        # for index,agent in enumerate(list(self.AgentsContains)):
        #     if index <self.num_pursuit:
        #         self.AgentsContains[agent].state.x = 5
        #         self.AgentsContains[agent].state.y = 5
        #         self.AgentsContains[agent].state.theta = math.pi/4
        #     else:
        #         self.AgentsContains[agent].state.x = 30
        #         self.AgentsContains[agent].state.y = 30
        #         self.AgentsContains[agent].state.theta = math.pi/4

        return self.AgentsContains

    "执行动作的函数"
    def step(self,action):
        next_obs_n = self.process(action)
        rew_n,done_n= self.reward()

        return next_obs_n ,rew_n,done_n

    "可视化函数,可视化暂时只支持可视化一个episode的场景"
    def render(self,fig,update,len,init):
        "len 是数据的长度"
        ani = animation.FuncAnimation(fig=fig, func=update, frames=len, interval=self.args.display_speed,init_func=init)
        ani.save('./picture/fig.gif')
        plt.show()

    "进行状态的迭代"
    def process(self,action):
        agents = self.AgentsContains
        "action 采用字典类型,{'name':[0,0,0,0]} 第一维占位符,无具体意义,第二维 线速度大小 第三维减第四维角速度大小"
        for index,key in enumerate(agents.keys()):
            "Keep theta between -pi and +pi "
            agents[key].state.theta = agents[key].state.theta - 2 * math.pi * np.floor(
                (agents[key].state.theta + math.pi) / (2 * math.pi))

            "速度"
            agents[key].state.v = action[index][1]
            agents[key].state.w = action[index][2]-action[index][3]

            "坐标转换"
            agents[key].state.x += math.cos(agents[key].state.theta) * agents[key].state.v * agents[key].max_v * self.args.timestep
            agents[key].state.y += math.sin(agents[key].state.theta) * agents[key].state.v * agents[key].max_v * self.args.timestep
            "偏航角变换"
            agents[key].state.theta += agents[key].state.w * agents[key].max_w * self.args.timestep

            "再次确定偏航角在-pi 到 pi之间"
            agents[key].state.theta = agents[key].state.theta - 2 * math.pi * np.floor(
                (agents[key].state.theta + math.pi) / (2 * math.pi))
            "边界碰撞检测处理,阻塞在边界位置"
            if agents[key].state.x < 0 or agents[key].state.x > self.args.map_width or agents[key].state.y < 0 or \
                    agents[key].state.x > self.args.map_height:
                agents[key].state.x = min(max(agents[key].state.x, 0.0), self.args.map_width)
                agents[key].state.y = min(max(agents[key].state.y, 0.0), self.args.map_height)
        return self.AgentsContains

    def reward(self):
        rewards = []
        dones = []

        for i in range(self.num_pursuit):
            done = False

            keys = list(self.AgentsContains.keys())
            pos_theta = relative_angle(self.AgentsContains[keys[i]],self.AgentsContains[keys[self.num_pursuit]])

            diff_theta = abs(self.AgentsContains[keys[i]].state.theta - pos_theta)
            if math.sqrt((self.AgentsContains[keys[self.num_pursuit]].state.y - self.AgentsContains[keys[i]].state.y) ** 2 + (
                    self.AgentsContains[keys[self.num_pursuit]].state.x - self.AgentsContains[keys[i]].state.x) ** 2) < 0.3: #0.5用来训练
                done = True
            if done:
                reward = 4
            else:
                reward = 0.5 * self.AgentsContains[keys[i]].state.v * math.cos(diff_theta)

            rewards.append(reward)
            dones.append(done)

        for i in range(self.num_evasion):
            rewards.append(0)
            dones.append(False)
        return rewards, dones

