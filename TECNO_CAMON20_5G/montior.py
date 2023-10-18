import subprocess
import re
import matplotlib.pyplot as plt
from setting import *
from SurfaceFlinger.get_fps import SurfaceFlingerFPS
# 创建两个空的列表来存储数据
time_data = []
current_usage_data = []
current_usage_data_avg = []
fps_data = []
# 设置图形窗口
plt.ion()
fig,(ax,ax2) = plt.subplots(1,2,sharey=False,sharex=False)
line1, = ax.plot(time_data, current_usage_data ,label='current realtime')
line2, = ax.plot(time_data, current_usage_data_avg, label='current avge')
line3, = ax2.plot(time_data,fps_data,label='fps')

# ax.set_ylim(0, 831600)
ax.set_title('current ')
ax.set_xlabel('Time (s)')
ax.set_ylabel('current (mA)')

ax.legend()
plt.grid(True)
# plt.yticks(range(300, 1500,20))
# 循环获取并显示current利用率





from PowerLogger.dimensity_8050_power import PowerLogger
power = PowerLogger(PHINE_IP,PHINE_PORT)
time = 1
time_window = 420
sum_avg = 0
Fps = SurfaceFlingerFPS(PHINE_IP,PHINE_PORT,TARGET_APP)
while True:

    current_usage_value = power.getPower()
    time += 1
    time_data.append(time)
    current_usage_data.append(power.getPower())
    if len(current_usage_data) <= time_window:
        sum_avg += current_usage_value
        current_usage_data_avg.append(sum_avg/len(current_usage_data))
    else:
        sum_avg = (sum_avg + current_usage_value - current_usage_data[len(current_usage_data)-time_window-1])
        current_usage_data_avg.append(sum_avg/time_window)
        
    
    if time > time_window:
        ax.set_xlim(time - time_window, time)
    fps_data.append(Fps.get_fps())
    line1.set_data(time_data, current_usage_data)
    line2.set_data(time_data, current_usage_data_avg)
    line3.set_data(time_data,fps_data)
    ax.relim()
    ax.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    
    plt.draw()
        
    plt.pause(0.0001)