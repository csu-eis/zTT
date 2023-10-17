import subprocess
import sys
sys.path.append(".")

from setting import *
# gpu_clock_list=[180000000, 267000000, 355000000, 430000000]
gpu_clock_list= [886000, 879000, 873000, 867000, 861000, 854000, 848000, 842000, 836000, 825000, 815000, 805000, 795000, 785000, 775000, 765000, 755000, 745000, 735000, 725000, 715000, 705000, 695000, 685000, 675000, 654000, 634000, 614000, 593000, 573000, 553000, 532000, 512000, 492000, 471000, 451000, 431000, 410000, 390000, 370000, 350000]
dir_thermal='/sys/devices/virtual/thermal'

class GPU:
    def __init__(self,ip,port):
        self.clk=3
        self.clock_data=[]
        self.temp_data=[]
        self.ip = f"{ip}:{port}"
        
        # fname='/sys/class/kgsl/kgsl-3d0/devfreq/max_freq'
        # subprocess.check_output(['adb', '-s', self.ip, 'shell', 'su -c', '\"echo', str(gpu_clock_list[3])+" >", fname+"\""])
        # fname='/sys/class/kgsl/kgsl-3d0/devfreq/min_freq'
        # subprocess.check_output(['adb', '-s', self.ip, 'shell', 'su -c', '\"echo', str(gpu_clock_list[0])+" >", fname+"\""])
		
    def setGPUclock(self,i):
        self.clk=i
        # fname='/sys/class/kgsl/kgsl-3d0/devfreq/userspace/set_freq'
        command =  f'adb -s {self.ip} shell "echo  {gpu_clock_list[i]} > /proc/gpufreq/gpufreq_opp_freq" '
        subprocess.check_output(command)
		
    def getGPUtemp(self):
       
        command = f'adb -s {self.ip} shell "cat /sys/class/thermal/thermal_zone4/temp"'
        output = subprocess.check_output(command)
        output = output.decode('utf-8')
        output = output.strip()
        # print(int(output)/1000)
        return int(output)/1000

    def getGPUclock(self):
        commad = f'adb -s {self.ip} shell "cat /proc/gpufreq/gpufreq_var_dump"'
        output = subprocess.check_output(commad)
        output = output.decode('utf-8')
        output = output.strip()
        # start_index = output.find("freq = ")
        # end_index = output.find(",", start_index)

        # 提取 freq 的值
        
        
        
        start_index = output.find("freq: ") + len("freq: ")
        end_index = output.find(",", start_index)
        freq = int(output[start_index:end_index])
        
       
        # print(freq)
  
        return int(freq)/1000000

    def collectdata(self):
        self.clock_data.append(self.getGPUclock())
        self.temp_data.append(self.getGPUtemp())
        # print(self.clock_data)
        # print(self.temp_data)

    def setUserspace(self):
        pass
    
        # fname='/sys/class/kgsl/kgsl-3d0/devfreq/governor'
        # subprocess.check_output(['adb', '-s', self.ip, 'shell',  'su -c', '\"echo userspace >', fname+"\""])
        # print('[gpu]Set userspace')
    
    def setdefault(self):
        # fname='/sys/class/kgsl/kgsl-3d0/devfreq/governor'
        # subprocess.check_output(['adb', '-s', self.ip, 'shell',  'su -c', '\"echo msm-adreno-tz >', fname+"\""])
        # print('[gpu]Set msm-adreno-tz')
        try: 
            commad = f'adb -s {self.ip} shell "echo 0 > /proc/gpufreq/gpufreq_opp_freq"'  #using this command ,system return 1 (error code)
            print(commad)
            subprocess.check_output(commad)
        except:
            pass
    
    def getCurrentClock(self):
        # fname='/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq'
        # output = subprocess.check_output(['adb', '-s', self.ip, 'shell',  'su -c', '\"cat', fname+"\""])
        # output = output.decode('utf-8')
        # output = output.strip()
        return self.getGPUclock()
        # print('[gpu]{}Hz'.format(output))
    def gpu_print(self):
        while 1:
            print(self.getCurrentClock())
if __name__ == "__main__":
    gpu = GPU(PHINE_IP,PHINE_PORT)
    gpu.setGPUclock(7)
    gpu.getGPUclock()
    # gpu.collectdata()
    # gpu.collectdata()
    # gpu.setdefault()
    gpu.getGPUclock()
    gpu.setGPUclock(len(gpu_clock_list)-1)
    gpu.getGPUclock()
    gpu.gpu_print()
    # gpu.getGPUclock()
    # gpu.getGPUclock()
    # gpu.getGPUclock()
    # gpu.setdefault()
    # i = 300000
    # while i :
    #     gpu.getGPUclock()
    #     i = i - 1
