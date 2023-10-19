# work in \TECNO_CAMON20_5G 
import sys
from perf_monitor.surfaceflinger.fps import SurfaceFlingerFPS
from perf_monitor.power.dimensity_8050_power import PowerLogger
from perf_monitor.cpu.dimensity_8050_cpu import CPU
from perf_monitor.gpu.dimensity_8050_gpu import GPU
from setting import *

if __name__ == "__main__" :
    cpu0 = CPU(0,'l',PHINE_IP,PHINE_PORT)
    cpu4 = CPU(4,'m',PHINE_IP,PHINE_PORT)
    cpu7 = CPU(7,'b',PHINE_IP,PHINE_PORT)
    gpu = GPU(PHINE_IP,PHINE_PORT)
    power = PowerLogger(PHINE_IP,PHINE_PORT)
    sf_fps_driver = SurfaceFlingerFPS(PHINE_IP,PHINE_PORT, keyword=TARGET_APP)
    while 1:
        print("cpu0 clock: ",cpu0.getCPUclock() ,"cpu4 clock: ",cpu4.getCPUclock() ,"cpu7 clock: ",cpu4.getCPUclock(),"cpu temp",cpu0.getCPUtemp(),"fsp ",sf_fps_driver.get_fps(),"power " ,power.getPower())