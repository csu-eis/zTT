#!/usr/bin/env python3
import os
import time
import cv2
import socket
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
import struct
import math
import sys
from setting import *
from SurfaceFlinger.get_fps import SurfaceFlingerFPS
from PowerLogger.dimensity_8050_power import PowerLogger
from CPU.dimensity_8050_cpu import CPU
from GPU.dimensity_8050_gpu import GPU

def save_csv(fps_data,c0,c4,c7,g,csv_path = CSV_PATH):
    pass
    
    f=open(csv_path,'w',encoding='utf-8',newline='')
    wr=csv.writer(f)
    wr.writerow(pl.power_data[1:])
    f.close()

    f=open(csv_path,'w',encoding='utf-8',newline='')
    wr=csv.writer(f)
    wr.writerow(c0.temp_data)
    wr.writerow(c4.temp_data)
    wr.writerow(c7.temp_data)
    wr.writerow(g.temp_data)
    f.close()

    f=open(csv_path,'w',encoding='utf-8',newline='')
    wr=csv.writer(f)
    wr.writerow(c0.clock_data)
    wr.writerow(c4.clock_data)
    wr.writerow(c7.clock_data)
    wr.writerow(g.clock_data)
    f.close()

    f=open(csv_path,'w',encoding='utf-8',newline='')
    wr=csv.writer(f)
    wr.writerow(fps_data)
    f.close()
def zTT_plot(fps_data,c0,c4,c7,g,pl,target_fps):
    ts = range(0, len(fps_data))
    fig = plt.figure(figsize=(12, 14))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.set_xlabel('time')
    ax1.set_ylabel('power(mw)')
    ax1.set_ylim([0, 4000]) 
    ax1.grid(True)
    ax1.plot(ts,pl.power_data[1:],label='TOTAL')
    ax1.legend(loc='upper right')
    ax1.set_title('Power')

    ax2.set_xlabel('time')
    ax2.set_ylabel('temperature')
    ax2.set_ylim([0, 70])
    ax2.grid(True)
    ax2.plot(ts,c0.temp_data,label='LITTLE')
    ax2.plot(ts,c4.temp_data,label='MIDDLE')
    ax2.plot(ts,c7.temp_data,label='BIG')
    ax2.plot(ts,g.temp_data,label='GPU')
    ax2.legend(loc='upper right')
    ax2.set_title('temperature')

    ax3.set_xlabel('time')
    ax3.set_ylabel('clock frequency(MHz)')
    ax3.set_ylim([0, 2000])
    ax3.grid(True)
    ax3.plot(ts,c0.clock_data,label='LITTLE')
    ax3.plot(ts,c4.clock_data,label='MIDDLE')
    ax3.plot(ts,c7.clock_data,label='BIG')
    ax3.plot(ts,g.clock_data,label='GPU')
    ax3.legend(loc='upper right')
    ax3.set_title('clock')

    ax4.set_xlabel('time')
    ax4.set_ylabel('FPS')
    ax4.set_ylim([0, 61])
    ax4.grid(True)
    ax4.plot(ts,fps_data,label='Average FPS')
    ax4.axhline(y=target_fps, color='r', linewidth=1)
    ax4.legend(loc='upper right')
    ax4.set_title('fps')
    # plt.title("client")
    plt.show()

if __name__=="__main__":

    ''' 
        Set initial state
        c_c -> CPU clock
        g_c -> GPU clock
        c_t -> CPU temperature
        g_t -> GPU temperature
        c_p -> power
    '''
    
    app = TARGET_APP
    server_ip = SERVER_IP
    server_port = SERVER_PORT
    pixel_ip = PHINE_IP
    pixel_port = PHINE_PORT
    target_fps = TARGET_FPS
    experiment_time = EXPERIMENT_TIME
  
    c0=CPU(0, cpu_type='l', ip=PHINE_IP, port=PHINE_PORT)
    c4=CPU(4, cpu_type='m', ip=PHINE_IP, port=PHINE_PORT)
    c7=CPU(7, cpu_type='b', ip=PHINE_IP, port=PHINE_PORT)
    g=GPU(PHINE_IP,PHINE_PORT)
    pl=PowerLogger(PHINE_IP,PHINE_PORT)
    
    # sf_fps_driver = SurfaceFlingerFPS(PHINE_IP,PHINE_PORT, keyword="org.videolan.vlc")
    sf_fps_driver = SurfaceFlingerFPS(PHINE_IP,PHINE_PORT, keyword=TARGET_APP)
    fps_data=[]
    c_c0=8
    c_c4=8
    c_c7=8
    g_c=3
    c_t=[]
    g_t=[]
    c_p=[]
    g_p=[]
    state=(c_c0,c_c4,c_c7,g_c,int(pl.getPower()/100),0, c_t, g_t)
    
    print("start successful")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port)) 
    print("connect successful")
    time.sleep(4)

    print("Start learning")
    iter = 0
    for t in range(experiment_time) :
        fps = float(sf_fps_driver.get_fps())
        if fps > 60:
            fps = 60.0
            
        fps_data.append(fps)
        c0.collectdata()
        c4.collectdata()
        c7.collectdata()
        g.collectdata()
        
        c_p.append(int(pl.getPower()) )
        c_t.append(float(c0.getCPUtemp()))
        g_t.append(float(g.getGPUtemp()))
        g_p.append(0)
        # 
        

        if iter>=4:
            next_state=(c_c0,c_c4,c_c7, g_c, np.average(np.asanyarray(c_p)), np.average(np.asanyarray(g_p)), np.average(np.asanyarray(c_t)), np.average(np.asanyarray(g_t)), np.average(np.asanyarray(fps)))
            # c_c: CPU clock g_c: GPU cock c_p: power g_p: ? c_t: CPU_temp g_t :gpu_temp fps
            send_msg=str(c_c0)+','+str(c_c4)+','+str(c_c7)+','+str(g_c)+','+str(np.average(np.asanyarray(c_p)))+','+str( np.average(np.asanyarray(g_p)))+','+str(np.average(np.asanyarray(c_t)))+','+str(np.average(np.asanyarray(g_t)))+','+str(np.average(np.asanyarray(fps)))
            print("clinet send")
            client_socket.send(send_msg.encode())
            print('[{}] state:{} next_state:{} fps:{}'.format(t, state, next_state, fps))
            state=next_state
            
            
            # get action
            recv_msg=client_socket.recv(SERVER_PORT).decode()
            clk=recv_msg.split(',')

            c_c0=int(clk[0])
            c_c4=int(clk[1])
            c_c7=int(clk[2])
            g_c=int(clk[3])

            c0.setCPUclock(c_c0)
            c4.setCPUclock(c_c4)
            c7.setCPUclock(c_c7)
            g.setGPUclock(g_c)
            iter=0
            
            
        iter+=1
        time.sleep(0.5)

    # Logging results
    print('Average Total power={} mW'.format(sum(pl.power_data)/len(pl.power_data)))
    print('Average fps = {} fps'.format(sum(fps_data)/len(fps_data)))
    
    save_csv(fps_data,c0,c4,c7,g,csv_path = CSV_PATH)
    
    # Plot results
    zTT_plot(fps_data,c0,c4,c7,g,pl,target_fps)
   



