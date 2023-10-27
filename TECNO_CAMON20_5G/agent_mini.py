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
def random_action():
    return np.array([random.randint(0,2),random.randint(0,2),random.randint(0,2)])
def limit_cpu_frq(x):
    if x < 0: x = 0
    elif x > 15: x = 15
    return x

    
def get_reward(old_fps,fps,target_fps,action):
    
    # u = 0
    # if gap < 1:
    #     u = 8
    # elif gap < 1.5:
    #     u = 0.5
    # elif gap < 2.5:
    #     u = 0.1
    gap = abs(fps - target_fps)
    # u = 1
    # gap = abs(fps - target_fps)
    # u = math.exp(-(gap-0.5)*0.5) + 1/(4*gap+0.5) - 1
    u = math.exp(-(gap-3))+ 1/(gap+0.1)
    # if fps < target_fps and curr_state.sum() > 3:
    #     u = -2
    # elif fps > target_fps and curr_state.sum() < 3:
    #     u = -2
    # elif fps == target_fps and curr_state[0] == 0 and curr_state[1] ==0 and curr_state[2] == 0:
    #     u = 10
    # elif fps == target_fps and (curr_state.sum() != 3  ):
    #     u = -2
    # else:
    #     u = 1

    # if abs(old_fps- target_fps) ==  abs(fps - target_fps) and abs(fps - target_fps) != 0 :
    #     return 0.1
    # elif abs(old_fps- target_fps) ==  abs(fps - target_fps) and abs(fps - target_fps) == 0:
    #     return 15
    # elif abs(fps - target_fps) < 0.5 :
    #     return 10
 
    # elif abs(old_fps - target_fps) < 0.5  and abs(fps - target_fps) > 1:
    #     if action[0] == 1 and action[1] == 1 and action[2] == 1:
    #         return 2
    #     return -20
    # elif abs(old_fps- target_fps) >  abs(fps - target_fps):
    #     return 5
    # else :
    #     return -10
    
    if abs(old_fps - target_fps) < 1 and  abs(old_fps - target_fps) < 1:
        return 8+math.exp(-(abs(fps - target_fps)))*6
    
    elif abs(old_fps - target_fps) == abs(fps- target_fps) and abs(fps- target_fps) >= 1:
        return -1
    elif abs(old_fps - target_fps) > abs(fps- target_fps) :
        return round((abs(old_fps - target_fps) - abs(fps- target_fps))*10,4)
    elif abs(old_fps - target_fps) < abs(fps- target_fps):
        return round(-(abs(fps- target_fps) - abs(old_fps - target_fps))*10,4)
    
    
    return 100000
def DQN_train(agent,server_socket,client_socket,epch,target,th = 0.60,sync_time = 15,train_n_batch = 5,batch_size = 16,inference_time = 50):
    i = 0
    t = 0
    old_fps = 0
    old_target = 0
    fps = 0
    prev_state=np.zeros(S_DIM,dtype=np.float32)
    curr_state = prev_state
    action = np.ones(ACTION_SPACE,dtype=np.float32)
    target_fps = int(random.sample(target,k=1)[0])
    arrived_dict = {key: 0 for key in target}
    while i <  epch:
        old_fps = fps
        # old_target = target_fps
        c_c0, c_c4, c_c7,fps =  get_data(client_socket)
        prev_state=curr_state
        gap = abs(fps- target_fps)
        if fps > target_fps:
            
            curr_state = np.array([c_c0,c_c4,c_c7,gap,1,0,0],dtype=np.float32)
        elif fps < target_fps:
            curr_state = np.array([c_c0,c_c4,c_c7,gap,0,1,0],dtype=np.float32)
        else:
            curr_state = np.array([c_c0,c_c4,c_c7,gap,0,0,1],dtype=np.float32)
        
        
        reward = get_reward(old_fps,fps,target_fps,action)

        agent.mem.push(prev_state[-4:], action,curr_state[-4:], reward)
        logger.info(f"mem prev_state {prev_state[0]} {prev_state[1]} {prev_state[2]} action {action[0]-1} {action[1]-1} {action[2]-1} cur_state {curr_state[0]} {curr_state[1]} {curr_state[2]} old_fps {old_fps} fps{fps} reward {reward}")
        agent.train(train_n_batch,batch_size,GAMMA,ALPHA)
        if abs(fps- target_fps) <= 0.5: 
            arrived_dict[target_fps] = arrived_dict[target_fps] + 1
            if arrived_dict[target_fps] > 5:
                arrived_dict[target_fps] = 0
                target_fps = int(random.sample(target,k=1)[0])
            logger.critical("arrived")
            
        
        if np.random.rand() <= th :
            if fps > target_fps:
                x = np.array([c_c0,c_c4,c_c7],np.float32)
                c_c0= limit_cpu_frq(c_c0 + random.randint(0,1))
                c_c4= limit_cpu_frq( c_c4 + random.randint(0,1))
                c_c7= limit_cpu_frq( c_c7 + random.randint(0,1))
                action = np.array([c_c0,c_c4,c_c7]) - x  + 1
                logger.debug(f"fps:{fps} reward{round(reward,4)} target_fps : {target_fps} curr :{int(curr_state[0])} {int(curr_state[1])} {int(curr_state[2])}  lower explore {c_c0}  {c_c4} {c_c7} ")
            else:
                x = np.array([c_c0,c_c4,c_c7],np.float32)
                c_c0= limit_cpu_frq(c_c0 - random.randint(0,1))
                c_c4= limit_cpu_frq( c_c4 - random.randint(0,1))
                c_c7= limit_cpu_frq( c_c7 - random.randint(0,1))
                action = np.array([c_c0,c_c4,c_c7]) - x  + 1
                logger.debug(f"fps:{fps} reward{round(reward,4)} target_fps : {target_fps} curr :{int(curr_state[0])} {int(curr_state[1])} {int(curr_state[2])}  higher explore {c_c0}  {c_c4} {c_c7} ")
        else:
            action=np.array(agent.max_action(torch.from_numpy(curr_state[-4:])))
            
            c_c0= limit_cpu_frq(c_c0 + action[0] - 1)
            c_c4= limit_cpu_frq(c_c4 + action[1] - 1)
            c_c7= limit_cpu_frq(c_c7 + action[2] - 1)
            logger.info (f"fps:{fps} reward{round(reward,4)} target_fps : {target_fps} curr :{int(curr_state[0])} {int(curr_state[1])} {int(curr_state[2])}  infernce       {c_c0}  {c_c4} {c_c7}  action {action[0]} {action[1]} {action[2]}")
            
        
        
        if t%sync_time==0:     
            agent.sync_model()
        
        # action = np.array([1,2,0],dtype=np.float32)
        send_msg=str(c_c0)+','+str(c_c4)+','+str(c_c7)+','+str(0) 
        client_socket.send(send_msg.encode())
        if t%inference_time == 0:
            DQN_inference(client_socket)
        t = t + 1
            
def DQN_inference(client_socket):
    
    temp_target_fps = int(random.sample(TARGET_FPS,k=1)[0])
    logger.warning(f"Inference target_fps {temp_target_fps}--------------------------------------------------")
    for i in range(30):
        c_c0, c_c4, c_c7,fps =  get_data(client_socket)
        gap = abs(fps - temp_target_fps)
        if fps > temp_target_fps:
            
            curr_stat = np.array([c_c0,c_c4,c_c7,gap,1,0,0],dtype=np.float32)
        elif fps < temp_target_fps:
            curr_stat = np.array([c_c0,c_c4,c_c7,gap,0,1,0],dtype=np.float32)
        else:
            curr_stat = np.array([c_c0,c_c4,c_c7,gap,0,0,1],dtype=np.float32)
      
        action=agent.max_action(torch.from_numpy(curr_stat[-4:]))
       
        c_c0= limit_cpu_frq(c_c0 + action[0] - 1)
        c_c4= limit_cpu_frq(c_c4 + action[1] - 1)
        c_c7= limit_cpu_frq(c_c7 + action[2] - 1)
        print(f"fps {fps} action {action[0]} {action[1]} {action[2]}  {c_c0} {c_c4} {c_c7}")
        send_msg=str(c_c0)+','+str(c_c4)+','+str(c_c7)+','+str(0) 
        client_socket.send(send_msg.encode())
    
def get_data(client_socket):
    try:
        msg = client_socket.recv(512).decode()
        state_tmp = msg.split(',')
        if not msg:
            logger.error('No receiveddata')
            return -1
    finally:
        c_c0 = int(state_tmp[0])
        c_c4 = int(state_tmp[1])
        c_c7 = int(state_tmp[2])
        fps= round(float(state_tmp[3]),2) # 取整数
    return c_c0,c_c4,c_c7,fps
if __name__ == "__main__" :
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("", PORT))
    server_socket.listen(5)
    
    agent = DQN_AGENT_AB(s_dim=S_DIM,h_dim=H_DIM,branches=BRANCHES,buffer_size=BUFFER_SIZE,params=None)
    # agent.load_model(SAVE_PATH)
    client_socket, address = server_socket.accept()
    DQN_train(agent,server_socket,client_socket,epch = EXPERIMENT_TIME,target=TARGET_FPS)
    
    