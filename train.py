import numpy as np
import math
from pursuit_env import Env
import maddpg.common.tf_util as U
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

import time


def parse_args():
    parser = argparse.ArgumentParser("parse args for multiagent environments")
    # Train
    parser.add_argument("--episode_len", type=int, default=500, help="step per episode")
    parser.add_argument("--model_save_rate", type=int, default=50, help="model save rete")
    parser.add_argument("--save_dir", type=str, default="./model/", help="model save dir")
    parser.add_argument("--load_model", type=bool, default=True, help="load model")
    parser.add_argument("--display", type=bool, default=True, help="display gif")
    parser.add_argument("--display_speed", type=int, default=20, help="gif speed per frame")

    # Agent
    parser.add_argument("--agent_maxspeed", type=float, default=3.0, help="max speed of agent")
    parser.add_argument("--agent_max_w", type=float, default=1.0, help="max w of agent")

    # MADDPG
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount")
    parser.add_argument("--batch_size", type=int, default=64, help="number of batch per update")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--input_state_dim", type=int, default=5, help="the input dim of agent actor net")
    parser.add_argument("--action_dim", type=int, default=4, help="action dim of agent")

    # Environment
    parser.add_argument("--timestep", type=float, default=0.1, help="timestep")
    parser.add_argument("--map_width", type=int, default=50, help="width of env")
    parser.add_argument("--map_height", type=int, default=50, help="height of env")
    parser.add_argument("--num_pursuit", type=int, default=4, help="number of pursuit")
    parser.add_argument("--num_evasion", type=int, default=1, help="number of evasion")
    parser.add_argument("--num_obstacle", type=int, default=0, help="number of obstacle")

    return parser.parse_args()


"计算相对距离"
def relative_dis(start_point, end_point):
    "start_point 和 end_point 都是智能体对象"
    return math.sqrt((start_point.state.x - end_point.state.x) ** 2 + (start_point.state.y - end_point.state.y) ** 2)

def relative_angle(start_point,end_point):
    "start_point 和 end_point 都是智能体对象"
    x_diff = end_point.state.x - start_point.state.x
    y_diff = end_point.state.y - start_point.state.y
    relative_theta = math.atan(y_diff / (x_diff + 0.0001))

    if y_diff > 0 and x_diff < 0:
        relative_theta += math.pi
    elif y_diff < 0 and x_diff < 0:
        relative_theta -= math.pi
    return relative_theta

"构建actor网络的输入"
def input_state(obs,num_pur):
    "num_agent 是追捕者和逃跑者的数量之和"
    input_states = []

    for agent in list(obs.keys())[:num_pur]:
        obs_tmp = []
        indexs = list(obs.keys())[:num_pur]
        evasion_indexs = list(obs.keys())[num_pur:]
        indexs.remove(agent)
        obs_1 = [obs[agent].state.theta,obs[agent].state.v,obs[agent].state.w]
        obs_2 = [[relative_angle(obs[agent],obs[index]),relative_dis(obs[agent],obs[index])] for index in indexs]
        obs_3 = [[relative_angle(obs[agent],obs[index]),relative_dis(obs[agent],obs[index])/env.args.map_width] for index in evasion_indexs]

        obs_tmp.append(list(np.array(obs_1).ravel()))
        # obs_.append(list(np.array(obs_temp2).ravel()))
        obs_tmp.append(list(np.array(obs_3).ravel()))
        # obs_tmp.append(list(np.array(obs_temp4).ravel()))

        # 降维
        temp_list = str(obs_tmp)
        temp_list = temp_list.replace('[', '')
        temp_list = temp_list.replace(']', '')
        obs_tmp = list(eval(temp_list))
        obs_tmp = np.array(obs_tmp)

        input_states.append(obs_tmp)

    for index in list(obs.keys())[num_pur:]:
        eva_obs = np.array(
            [obs[index].state.x, obs[index].state.y, obs[index].state.theta, obs[index].state.v, obs[index].state.w])
        input_states.append(eva_obs)

    return input_states

"计算切线的函数 跟可视化有关"
def tangent_line(x0,y0,k):
    xs = np.linspace(x0, x0 + math.cos(k), 100)
    ys = y0 + math.tan(k) * (xs - x0)
    return xs,ys
"更新  跟可视化有关"
def update(frame):
    for index,key in enumerate(list(env.AgentsContains.keys())):
        xs, ys = tangent_line(x_lists[index][frame], y_lists[index][frame], theta_lists[index][frame])
        tangent_anis[index].set_data(xs, ys)
        ln[index][0].set_data(x_lists[index][frame], y_lists[index][frame])

"初始化画面刻度 跟可视化有关"
def init():
    ax.set_xlim(0, args.map_width)
    ax.set_ylim(0, args.map_height)


if __name__ == '__main__':
    with U.single_threaded_session():
        args = parse_args()
        env = Env(args)
        obs = env.reset()
        train_step = 1
        episode = 0
        saver = tf.train.Saver()
        agents = env.AgentsContains

        U.initialize()

        if args.load_model:
            print('Loading previous state...')
            U.load_state(args.save_dir)

        x_lists=[[] for i in range(args.num_pursuit+args.num_evasion+args.num_obstacle)]
        y_lists=[[] for i in range(args.num_pursuit+args.num_evasion+args.num_obstacle)]
        theta_lists=[[] for i in range(args.num_pursuit+args.num_evasion+args.num_obstacle)]
        episode_rew = [[] for i in range(args.num_pursuit+args.num_evasion)]

        "可视化的相关初始化"
        fig, ax = plt.subplots()  #本行要在ln定义之前
        ln=[]
        for i in range(args.num_pursuit+args.num_evasion+args.num_obstacle):
            if i<args.num_pursuit:
                ln.append(plt.plot([], [], 'ro'))
            elif  i>=args.num_pursuit and  i<args.num_pursuit+args.num_evasion:
                ln.append(plt.plot([], [], 'bo'))
            else:
                ln.append(plt.plot([], [], 'go'))
        tangent_anis = [plt.plot(0, 0, c='blue', alpha=0.8)[0] for i in range(args.num_pursuit+args.num_evasion+args.num_obstacle)]



        print("Starting iterations...")
        while True:

            input_states = input_state(obs,args.num_pursuit)

            action_n = [env.AgentsContains[index].action(obs) for index, obs in zip(list(env.AgentsContains.keys())[:args.num_pursuit], input_states[:args.num_pursuit])]
            # action_n = [np.array([0,1,0,0]) for _ in range(env.num_pursuit)]
            for _ in range(args.num_evasion):
                action_n.append(np.random.uniform(low=0, high=0, size=[4, ]))

            next_obs_n, rew_n, done_n = env.step(action_n)

            next_input_states = input_state(next_obs_n,args.num_pursuit)

            terminal = (train_step % args.episode_len == 0)

            for i, key in enumerate(agents.keys()):
                agents[key].experience(input_states[i], action_n[i], rew_n[i], next_input_states[i], done_n[i], terminal)
                x_lists[i].append(agents[key].state.x)
                y_lists[i].append(agents[key].state.y)
                theta_lists[i].append(agents[key].state.theta)
                episode_rew[i].append(rew_n[i])

            input_states = next_input_states
            train_step += 1

            if terminal:
                episode += 1
                print("episode:{},reward{}".format(episode,[sum(episode_rew[i]) for i in range(args.num_pursuit+args.num_evasion)]))


                if args.display:
                    env.render(fig,update,args.episode_len,init)

                for i in range(len(agents)):
                    episode_rew[i].clear()
                    x_lists[i].clear()
                    y_lists[i].clear()
                    theta_lists[i].clear()

                obs = env.reset()
                input_states = input_state(obs,args.num_pursuit)




            for index in env.AgentsContains.keys():
                env.AgentsContains[index].preupdate()  # 清空replay_sample_index的值

            train_contains = [env.AgentsContains[index] for index in env.AgentsContains.keys()]

            for index in env.AgentsContains.keys():
                loss = env.AgentsContains[index].update(train_contains, train_step)


            # save model, display training output
            if (episode % args.model_save_rate == 0) and terminal:
                print('Now, Saving...')
                U.save_state(args.save_dir, saver=saver)

