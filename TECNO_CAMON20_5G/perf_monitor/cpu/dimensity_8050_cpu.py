import subprocess
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../..")
from setting import PHINE_IP,PHINE_PORT
little_cpu_clock_list = [2000000 ,1800000, 1725000, 1625000, 1525000, 1450000, 1350000, 1250000, 1075000 ,1000000, 925000, 850000, 750000, 675000, 600000, 500000]
middle_cpu_clock_list = [2600000, 2507000, 2354000, 2200000, 1985000, 1855000, 1740000, 1624000, 1537000, 1451000, 1335000, 1162000, 1046000, 902000, 700000, 437000]
big_cpu_clock_list =    [3000000, 2892000, 2713000, 2600000, 2463000, 2284000, 2141000, 1998000, 1820000, 1632000, 1482000, 1370000, 1258000, 1108000, 921000, 659000]

# little_cpu_clock_list.reverse()
# middle_cpu_clock_list.reverse()
# big_cpu_clock_list.reverse()

dir_thermal='/sys/devices/virtual/thermal'
cpu_cluster= {0: 0, 1:0, 2:0, 3:0 ,4:1, 5:1 ,6:1 ,7:2}
cpu_policy = {0: "policy0",1: "policy0",2: "policy0",3: "policy0",4: "policy4",5: "policy4",6: "policy4",7: "policy7"}
class CPU:
    def __init__(self,idx,cpu_type,ip,port):
        self.idx=idx
        self.cpu_type = cpu_type
        self.ip = f"{ip}:{port}"
        self.port = port
        self.clock_data=[]
        self.temp_data=[]

        if self.cpu_type == 'b':
            self.max_freq = 15
            self.clk = 0
            self.cpu_clock_list = big_cpu_clock_list
        elif self.cpu_type == 'm':
            self.max_freq = 15
            self.clk = 8
            self.cpu_clock_list = middle_cpu_clock_list
        elif self.cpu_type == 'l':
            self.max_freq = 15
            self.clk = 8
            self.cpu_clock_list = little_cpu_clock_list

        
    def setCPUclock(self,i):
        self.clk=i

        commad = f'adb -s {self.ip} shell "cat /proc/ppm/policy/ut_fix_freq_idx'
        out = subprocess.check_output(commad)
        numbers = re.findall(r'\d+', out.decode())
        numbers = [int(num) for num in numbers]


        numbers[cpu_cluster[self.idx]*2+1] = i

        commad = f'adb -s {self.ip} shell "echo {numbers[1]} {numbers[3]} {numbers[5]} > /proc/ppm/policy/ut_fix_freq_idx"'
        print(self.idx,commad)
        subprocess.check_output(commad)

        
    def getCPUtemp(self):
        fname="/sys/class/thermal/thermal_zone4/temp  "
        commad = f'adb -s {self.ip} shell "cat {fname}"'
        output = subprocess.check_output(commad)
        output = output.decode('utf-8')
        output = output.strip()
        # print(str(int(output)/1000))
        return int(output)/1000

    def getCPUclock(self):
        fname=f'/sys/devices/system/cpu/cpu{self.idx}/cpufreq/cpuinfo_cur_freq'
        commad  = f'adb -s {self.ip} shell "cat {fname}"'
        output = subprocess.check_output(commad)
        output = output.decode('utf-8')
        output = output.strip()
        # print(output)
        return int(output)/1000



    def collectdata(self):
        self.clock_data.append(self.getCPUclock())
        self.temp_data.append(self.getCPUtemp())
        # print(self.clock_data,self.temp_data)


    def setUserspace(self):
   
        commad = f'adb -s {self.ip} shell "echo userspace > /sys/devices/system/cpu/cpufreq/{cpu_policy[self.idx]}/scaling_governor" '
        subprocess.check_output(commad)
    @staticmethod
    def setgovernor(mode = "schedutil"):
        for i in [0,4,7]:
            commad = f'adb -s {PHINE_IP}:{PHINE_PORT} shell "echo {mode} > /sys/devices/system/cpu/cpufreq/policy{i}/scaling_governor" '
            subprocess.check_output(commad)
    
    @staticmethod
    def setdefault():
        commad = f'adb -s {PHINE_IP}:{PHINE_PORT} shell "echo -1 -1 -1 > /proc/ppm/policy/ut_fix_freq_idx"'
        subprocess.check_output(commad)
        CPU.setgovernor()
if __name__ == "__main__":
    print("main start \n")
    cpucontrel0 = CPU(0,'5',PHINE_IP,PHINE_PORT)
    cpucontrel4 = CPU(4,'5',PHINE_IP,PHINE_PORT)
    cpucontrel7 = CPU(7,'5',PHINE_IP,PHINE_PORT)
    cpucontrel0.setCPUclock(7)
    cpucontrel4.setCPUclock(7)
    cpucontrel7.setCPUclock(7)
    # cpucontrel0.getCPUtemp()
    # # cpucontrel0.getCPUclock(5)
    # cpucontrel0.collectdata()
    # cpucontrel4.collectdata()
    # cpucontrel7.collectdata()
    # cpucontrel7.setUserspace()
    # cpucontrel4.setUserspace()
    # CPU.setdefault()