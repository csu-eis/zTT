import subprocess
import re
little_cpu_clock_list = [2000000 ,1800000, 1725000, 1625000, 1525000, 1450000, 1350000, 1250000, 1075000 ,1000000, 925000, 850000, 750000, 675000, 600000, 500000]
middle_cpu_clock_list = [2600000, 2507000, 2354000, 2200000, 1985000, 1855000, 1740000, 1624000, 1537000, 1451000, 1335000, 1162000, 1046000, 902000, 700000, 437000]
big_cpu_clock_list =    [3000000, 2892000, 2713000, 2600000, 2463000, 2284000, 2141000, 1998000, 1820000, 1632000, 1482000, 1370000, 1258000, 1108000, 921000, 659000]
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
            self.max_freq = 9
            self.clk = 9
            self.cpu_clock_list = big_cpu_clock_list
        elif self.cpu_type == 'l':
            self.max_freq = 8
            self.clk = 8
            self.cpu_clock_list = little_cpu_clock_list
        elif self.cpu_type == 'm':
            self.max_freq
            self.clk
            self.cpu_clock_list = middle_cpu_clock_list
            
        
        
        # fname='/sys/devices/system/cpu/cpu%s/cpufreq/scaling_max_freq' %(self.idx)
        # subprocess.check_output(f'adb -s {self.ip}  shell "{self.cpu_clock_list[self.max_freq]} > fname"')
        
        # fname='/sys/devices/system/cpu/cpu%s/cpufreq/scaling_min_freq' %(self.idx)
        # subprocess.check_output(['adb', '-s', self.ip, 'shell', 'su -c', '\"echo', str(self.cpu_clock_list[0])+" >", fname+"\""])		
        
    def setCPUclock(self,i):
        self.clk=i

        # fname='/sys/devices/system/cpu/cpu%s/cpufreq/scaling_setspeed' %(self.idx)
        # commad = f'adb -s {self.ip} shell  "echo {self.cpu_clock_list[i]} "'
        # commad = f'adb -s {self.ip} shell "cat /proc/ppm/policy/ut_fix_freq_idx'
        # out = subprocess.check_output(commad)
        # print(out)
  
        commad = f'adb -s {self.ip} shell "cat /proc/ppm/policy/ut_fix_freq_idx'
        out = subprocess.check_output(commad)
        numbers = re.findall(r'\d+', out.decode())
        numbers = [int(num) for num in numbers]

        numbers[cpu_cluster[self.idx]*2+1] = i
        print(cpu_cluster[i],numbers)
        commad = f'adb -s {self.ip} shell "echo {numbers[1]} {numbers[3]} {numbers[5]} > /proc/ppm/policy/ut_fix_freq_idx"'
        print(commad)
        subprocess.check_output(commad)

        
    def getCPUtemp(self):
        fname="/sys/class/thermal/thermal_zone0/temp  "
        commad = f'adb -s {self.ip} shell "cat {fname}"'
        output = subprocess.check_output(commad)
        output = output.decode('utf-8')
        output = output.strip()
        print(str(int(output)/1000))
        return int(output)/1000

    def getCPUclock(self, idx):
        fname=f'/sys/devices/system/cpu/cpu{idx}/cpufreq/cpuinfo_cur_freq'
        commad  = f'adb -s {self.ip} shell "cat {fname}"'
        output = subprocess.check_output(commad)
        output = output.decode('utf-8')
        output = output.strip()
        print(output)
        return int(output)/1000

    # def getAvailableClock(self):
    # 	for i in range(0,8):
    # 		fname='/sys/devices/system/cpu/cpu%s/cpufreq/scaling_available_frequencies' %(i)
    # 		commad = f'adb -s {self.ip} shell "cat {fname}"'
    # 		output = subprocess.check_output(commad)
    # 		output = output.decode('utf-8')
    # 		output = output.strip()

    def collectdata(self):
        self.clock_data.append(self.getCPUclock(self.idx))
        self.temp_data.append(self.getCPUtemp())
        print(self.clock_data)
        print(self.temp_data)

    # def currentCPUstatus(self):
    # 	fname='/sys/devices/system/cpu/online'
    # 	with open(fname,'r') as f:
    # 		line=f.readline()
    # 		print(line)
    # 		f.close()

    # def getCurrentClock(self):
    # 	if self.cpu_type == 'l':
    # 		for i in range(0,4):
    # 			fname='/sys/devices/system/cpu/cpu%s/cpufreq/cpuinfo_cur_freq' %(i)
    # 			commad = f'adb -s {self.ip} shell "cat {fname}"'
    # 			output = subprocess.check_output(commad)
    # 			output = output.decode('utf-8')
    # 			output = output.strip()
    # 			print('[cpu{}]{}KHz '.format(i,output),end="")

    # 	elif self.cpu_type == 'm':
    # 		for i in range(5,6):
    # 			fname=f'/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_cur_freq' 
    # 			commad = f'adb -s {self.ip} shell "cat {fname}"'
    # 			output = subprocess.check_output(commad)
    # 			output = output.decode('utf-8')
    # 			output = output.strip()
    # 			print('[cpu{}]{}KHz '.format(i,output),end="")
    # 	elif self.cpu_type == 'b':
    # 		pass
            
    

    def setUserspace(self):
        # pass
        # if self.cpu_type == 'l':
        # 	for i in range(0,6):
        # 		fname='/sys/devices/system/cpu/cpu%s/cpufreq/scaling_governor' %(i)
        # 		commad = f'adb -s {self.ip} shell "cat {fname}"'
        # 		subprocess.check_output(['adb', '-s', self.ip, 'shell', 'su -c', '\"echo userspace >', fname+"\""])
        # 		print('[cpu{}]Set userspace '.format(i),end="")

        # if self.cpu_type == 'b':
        # 	for i in range(6,8):
        # 		fname='/sys/devices/system/cpu/cpu%s/cpufreq/scaling_governor' %(i)
        # 		subprocess.check_output(['adb', '-s', self.ip, 'shell', 'su -c', '\"echo userspace >', fname+"\""])
        # 		print('[cpu{}]Set userspace '.format(i),end="")

            
        commad = f'adb -s {self.ip} shell "echo userspace > /sys/devices/system/cpu/cpufreq/{cpu_policy[self.idx]}/scaling_governor" '
        subprocess.check_output(commad)

    def setdefault(self, mode):
        # if self.cpu_type == 'l':
        # 	for i in range(0,6):
        # 		fname='/sys/devices/system/cpu/cpu%s/cpufreq/scaling_governor' %(i)
        # 		subprocess.check_output(['adb', '-s', self.ip, 'shell', 'su -c', '\"echo '+mode+' >', fname+"\""])
        # 		print('[cpu{}]Set {}'.format(i,mode),end="")

        # if self.cpu_type == 'b':
        # 	for i in range(6,8):
        # 		fname='/sys/devices/system/cpu/cpu%s/cpufreq/scaling_governor' %(i)
        # 		subprocess.check_output(['adb', '-s', self.ip, 'shell', 'su -c', '\"echo '+mode+' >', fname+"\""])
        # 		print('[cpu{}]Set {}'.format(i,mode),end="")
          commad = f'adb -s {self.ip} shell "echo {mode} > /sys/devices/system/cpu/cpufreq/{cpu_policy[self.idx]}/scaling_governor" '
          subprocess.check_output(commad)

if __name__ == "__main__":
    print("main start \n")
    cpucontrel0 = CPU(0,'5',"172.16.101.94","43407")
    cpucontrel4 = CPU(4,'5',"172.16.101.94","43407")
    cpucontrel7 = CPU(7,'5',"172.16.101.94","43407")
    cpucontrel0.setCPUclock(7)
    cpucontrel7.setCPUclock(5)
    cpucontrel0.getCPUtemp()
    cpucontrel0.getCPUclock(5)
    cpucontrel0.collectdata()
    cpucontrel0.collectdata()
    cpucontrel0.collectdata()
    cpucontrel7.setUserspace()
    cpucontrel4.setUserspace()