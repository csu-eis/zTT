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
from pylogger import get_logger
PORT = SERVER_PORT
experiment_time=EXPERIMENT_TIME #14100
action_space=ACTION_SPACE
target_fps=TARGET_FPS
target_temp=TARGET_TEMP
is_training = IS_TRAIN
logger = get_logger("agent")
beta= 2 #4





    
def get_w(t,t_s):
    l_v = 0.1
    if t < t_s:
        return l_v * math.tanh(t_s - t)
    else:
        return -10 * l_v

def get_reward(fps, power, target_fps, c_t, g_t, c_t_s, g_t_s, beta):
    # print('power={}'.format(power))
    # u=max(1,fps/target_fps)

    w = get_w(c_t,c_t_s) + get_w(g_t,g_t_s)
    
    if (fps - target_fps) > 2:
        p = 1-math.exp(-500.0/power)
    else :
        p = 1
    
    u=math.exp(-abs((fps-target_fps))*0.3) + 1/(abs((fps-target_fps))+0.5)
    # return u*p
    return u
  

def get_reward_ztt(fps, power, target_fps, c_t, g_t, c_t_s, g_t_s, beta):
  
   
    if fps - target_fps >= 1:
        u = 1
    else :
        u = fps / target_fps
    
    # 先不考虑功耗项和温度项
    # u = u + beta/power
        
    return u 


    
def format_list(l):
    string = '['+','.join([f'{float(i):0.2f}' for i in l])+']'
    return string

def DQN_train():
    ...
def DQN_inferenc():
    ...
if __name__=="__main__":
    os.makedirs(f"save_model/fps_mixed/",exist_ok=True)

    agent = DQN_AGENT_AB(s_dim=11,h_dim=32,branches=[16,16,16,40],buffer_size=16000,params=None)
    agent.load_model(f"save_model/fps_mixed/")
    train_start = 2
    # scores, episodes = [], []

    t=0

    ts=[]
    fps_data=[]
    power_data=[]


    cnt=0
    c_c=16
    g_c=40
    c_t=60
    g_t=60

    logger.critical("TCPServr Waiting on port 8702")
    prev_state=np.asanyarray([1,0,1,0,1,0,0,0,0],dtype=np.float32)
    action=np.asanyarray([0,0,0,0],dtype=np.float32)
 
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("", PORT))
    server_socket.listen(5)
   
    train_count=0
    inference = 0
    curr_state = prev_state
    done = 0
    arrived_time = 0
    target_fps = 10
    try:
        client_socket, address = server_socket.accept()
    

        while t<experiment_time:
            # 获取到手机的state信息
            msg = client_socket.recv(512).decode()
            state_tmp = msg.split(',')
            
            if not msg:
                logger.error('No receiveddata')
                # print('No receiveddata')
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
            
            
            prev_state=curr_state

            curr_state=np.asanyarray([c_c0,c_c4,c_c7, g_c, c_p, g_p, c_t, g_t,fps,target_fps,done],dtype=np.float32)
            reward = get_reward(fps, c_p+g_p, target_fps, c_t, g_t, target_temp, target_temp, beta)
            if t > 2:
                logger.info('[{}] state:{}, action:{}, next_state:{}, reward:{:0.2f}, fps:{:0.2f}'.format(t, 
                            format_list(prev_state),
                            format_list(action),
                            format_list(curr_state),
                            reward,
                            fps))
                agent.mem.push(prev_state, action,curr_state, reward)
           

            if is_training and t!=0 :
                done = 0
                if len(agent.mem) >= train_start:
                    losses=agent.train(4,len(agent.mem)//4,BATCH_SIZE)
                    agent.save_model(train_count ,f"save_model/fps_{TARGET_FPS}/")
                    train_count+=1
                    logger.info(f"[Save model], losses:[{','.join([f'{i:0.2f}' for i in losses])}]",)
                    
                if abs(fps- target_fps) <= 1:
                    c_c0=int(random.randint(0,15))
                    c_c4=int(random.randint(0,15))
                    c_c7=int(random.randint(0,15))
                    g_c=int(random.randint(0,39))
                    done = 1
                    if arrived_time == ARRIVED_TIME:
                        target_fps = int(random.sample([10,14,18],k=1)[0])
                        arrived_time = 0
                    arrived_time = arrived_time + 1;
                    logger.critical(f"arrived target fps c_c0: {c_c0} c_c4: {c_c4} c_c7: {c_c7} g_c: {g_c} target_fps {target_fps} arrived_time {arrived_time}")
                    
                    
                elif np.random.rand() <= 0.3:
                    logger.info('previous clock : {} {} {} {}'.format(c_c0,c_c4,c_c7,g_c))
                    # NOTE CHECK THESE
                    if target_fps - fps > 0:  # explore higher space
                        c_c0=int(random.randint(max(0,c_c0-4),int(c_c0)))
                        c_c4=int(random.randint(max(0,c_c4-4),int(c_c4)))
                        c_c7=int(random.randint(max(0,c_c7-4),int(c_c7)))
                        g_c=int(random.randint(max(0,g_c-4),int(g_c)))
                        logger.info('explore higher clock@@@@@  {} {} {} {}'.format(c_c0,c_c4,c_c7,g_c))
                    elif fps - target_fps > 0:  # explore higher space
                        c_c0=int(random.randint(int(c_c0),min(15,c_c0+4)))
                        c_c4=int(random.randint(int(c_c4),min(15,c_c4+4)))
                        c_c7=int(random.randint(int(c_c7),min(15,c_c7+4)))
                        g_c=int(random.randint(int(g_c),min(39,g_c+6)))
                        logger.info('explore lower clock@@@@@  {} {} {} {}'.format(c_c0,c_c4,c_c7,g_c))

                else:
                    action=agent.max_action(torch.from_numpy(curr_state))
                    c_c0=action[0]
                    c_c4=action[1]
                    c_c7=action[2]
                    g_c=action[3]
                action = np.asanyarray([c_c0,c_c4,c_c7,g_c],dtype=np.float32)
              
               
                if train_count%8==0 and train_count!=0:
                    agent.sync_model()
                    logger.info("[Sync model]")

            #########
            else: # inference
                logger.critical(f"————————————————————————————————————————————————————————")
                logger.info(f" inference fps:{fps} c_c0: {c_c0} c_c4: {c_c4} c_c7: {c_c7} g_c: {g_c}")
                # if inference > 10:
                #     is_training = True
                #     inference = 0
                action=agent.max_action(torch.from_numpy(curr_state))
                c_c0=action[0]
                c_c4=action[1]
                c_c7=action[2]
                g_c=action[3]
                inference = inference + 1
                
            send_msg=str(c_c0)+','+str(c_c4)+','+str(c_c7)+','+str(g_c) 
            client_socket.send(send_msg.encode())
            
            t=t+1

    finally:
        server_socket.close()
    

    

    




