#!/usr/bin/env python3
import pickle
import warnings
import os
from threading import Thread
import time
import numpy as np
import random
import socket
import struct
import math
import sys
import torch
from torch import nn
from setting import *

from models.dqn_model import DQN_AGENT_AB
import matplotlib.pyplot as plt


PORT = SERVER_PORT
experiment_time=EXPERIMENT_TIME #14100
action_space=ACTION_SPACE
target_fps=TARGET_FPS
target_temp=TARGET_TEMP
beta= 2 #4





    
def get_w(t,t_s):
    l_v = 0.1
    if t < t_s:
        return l_v * math.tanh(t_s - t)
    else:
        return -10 * l_v

def get_reward(fps, power, target_fps, c_t, g_t, c_t_s, g_t_s, beta):
    print('power={}'.format(power))
    # u=max(1,fps/target_fps)

    w = get_w(c_t,c_t_s) + get_w(g_t,g_t_s)
    
    if fps > target_fps :
        p = -1*power/4500 * (fps - target_fps)/20
    else :
        p = 0
    
    u=math.exp(-abs((fps-target_fps))*0.3) + 1/(abs((fps-target_fps))+0.5)
    return u
   
def save_agent(anget,PICKLE_PATH):
    f = open(PICKLE_PATH, 'wb')
    pickle.dump(anget, f)
    f.close()
    print("对象持久化")
    
def load_agent(PICKLE_PATH):
    f = open(PICKLE_PATH, 'rb')
    object = pickle.load(f)
    print("load object from disk")
    f.close()
    return object 

     
if __name__=="__main__":
    os.makedirs("save_model/",exist_ok=True)

    agent = DQN_AGENT_AB(s_dim=9,h_dim=32,branches=[16,16,16,40],buffer_size=16000,params=None)
    agent.load_model("save_model/")
    train_start = 32
    # scores, episodes = [], []

    t=1

    ts=[]
    fps_data=[]
    power_data=[]


    cnt=0
    c_c=16
    g_c=40
    c_t=60
    g_t=60

    print("TCPServr Waiting on port 8702")
    prev_state=np.asanyarray([1,0,1,0,1,0,0,0,0],dtype=np.float32)
    score=0
    action=(0,0,0,0)
    copy=1

    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("", PORT))
    server_socket.listen(5)
    is_training = False
    n_round=0
    try:
        client_socket, address = server_socket.accept()
        fig = plt.figure(figsize=(6,7))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        # ax3 = fig.add_subplot(4, 1, 3)
        # ax4 = fig.add_subplot(4, 1, 4)

        while t<experiment_time:
            # 获取到手机的state信息
            msg = client_socket.recv(512).decode()
            state_tmp = msg.split(',')
            
            if not msg:
                print('No receiveddata')
                break

            c_c0 = int(state_tmp[0])
            c_c4 = int(state_tmp[1])
            c_c7 = int(state_tmp[2])
            g_c=int(state_tmp[3])
            c_p=float(state_tmp[4])
            g_p=float(state_tmp[5])
            c_t=float(state_tmp[6])
            g_t=float(state_tmp[7])
            fps=float(state_tmp[8])

            curr_state=np.asanyarray([c_c0,c_c4,c_c7, g_c, c_p, g_p, c_t, g_t,fps],dtype=np.float32)
            reward = get_reward(fps, c_p+g_p, target_fps, c_t, g_t, target_temp, target_temp, beta)
        
            # replay memory
            agent.mem.push(prev_state, action,curr_state, reward)

            print('[{}] state:{}, action:{}, next_state:{}, reward:{}, fps:{}'.format(t, prev_state,action,curr_state,reward,fps))
            fps_data.append(fps)
            power_data.append(c_p+g_p)
            ts.append(t)
            
#			
            prev_state=curr_state

            
            if is_training:
                if len(agent.mem) >= train_start:
                    agent.train(1,len(agent.mem)//16,16)
                    agent.save_model(n_round ,"save_model/")
                    n_round+=1
                    print("[Save model]")
                    
                if np.random.rand() <= 0.3 and fps<target_fps:
                    print('previous clock : {} {}'.format(c_c,g_c))
                    # NOTE CHECK THESE
                    c_c0=int(random.randint(0,int(c_c0)))
                    c_c4=int(random.randint(0,int(c_c4)))
                    c_c7=int(random.randint(0,int(c_c7)))
                    g_c=int(random.randint(0,int(g_c)))

                    print('explore higher clock@@@@@  {} {}'.format(c_c0,g_c))
                    print('explore higher clock@@@@@  {} {}'.format(c_c4,g_c))
                    print('explore higher clock@@@@@  {} {}'.format(c_c7,g_c))
                    action = (c_c0,c_c4,c_c7,g_c)

                    # action=3*int(c_c/3)+int(g_c)-1
                elif np.random.rand() <= 0.3 and fps>target_fps:
                    print('previous clock : {} {}'.format(c_c,g_c))
                    # NOTE CHECK THESE
                    c_c0=int(random.randint(int(c_c0),15))
                    c_c4=int(random.randint(int(c_c4),15))
                    c_c7=int(random.randint(int(c_c7),15))
                    g_c=int(random.randint(int(g_c),39))

                    print('explore lower clock@@@@@  {} {}'.format(c_c0,g_c))
                    print('explore lower clock@@@@@  {} {}'.format(c_c4,g_c))
                    print('explore lower clock@@@@@  {} {}'.format(c_c7,g_c))
                    action = (c_c0,c_c4,c_c7,g_c)
                else:
                    
                    action=agent.max_action(torch.from_numpy(curr_state))
                    c_c0=action[0]
                    c_c4=action[1]
                    c_c7=action[2]
                    g_c=action[3]

                if n_round%30==0 and n_round!=0:
                    agent.sync_model()
                    print("[Sync model]")
            else:
                action=agent.max_action(torch.from_numpy(curr_state))
                c_c0=action[0]
                c_c4=action[1]
                c_c7=action[2]
                g_c=action[3]

            send_msg=str(c_c0)+','+str(c_c4)+','+str(c_c7)+','+str(g_c)
            client_socket.send(send_msg.encode())
            
            
            
            ax1.plot(ts, fps_data, linewidth=1, color='red')
            ax1.axhline(y=target_fps, xmin=0, xmax=1000)
            ax1.set_title(f'Frame rate (Target fps = {TARGET_FPS}) ')
            ax1.set_ylabel('Frame rate (fps)')
            ax1.set_xlabel('Time (s) ')
            ax1.set_xticks([0, 200, 400, 600, 800,1000])
            ax1.set_yticks([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
            ax1.grid(True)

            ax2.plot(ts, power_data, linewidth=1, color='blue')
            ax2.set_title('Power consumption')
            ax2.set_ylabel('Power (mW)')
            ax2.set_yticks([0, 2000, 4000, 6000, 8000])
            ax2.set_xticks([0, 250, 500, 750, 1000])
            ax2.set_xlabel('Time (s) ')
            ax2.grid(True)
            plt.tight_layout()
            plt.pause(0.1)



            t=t+1

     


    finally:
        server_socket.close()
    
    # print(reward_tmp)
    # ts = range(0, len(avg_q_max_data))
    plt.figure(1)
    plt.xlabel('time')
    plt.ylabel('Avg Q-max')
    plt.grid(True)
    # plt.plot(ts,avg_q_max_data, label='avg_q_max')
    plt.legend(loc='upper left')
    plt.title('Average max-Q')
    plt.show()
    # plt.savefig("./p.png")
    
    

    




