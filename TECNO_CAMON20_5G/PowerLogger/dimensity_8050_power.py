import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from setting import PHINE_IP,PHINE_PORT
import subprocess
class PowerLogger:
    def __init__(self,ip,port):
        self.power=0
        self.ip = f"{ip}:{port}"
        self.voltage=0
        self.current=0
        self.bool_power_supply= 0 # 1 供电 0 不供电
        self.power_data = []
        self.voltage_data = []
        self.current_data = []


    def getVoltage(self):
        
        command = f'adb -s {self.ip} shell " cat /sys/class/power_supply/battery/voltage_now" '
        out  = subprocess.check_output(command).decode('utf-8')

        self.voltage = int(out)/1000
        self.voltage_data.append(self.voltage)

        return self.voltage

    def getCurrent(self):

        command = f'adb -s {self.ip} shell " cat /sys/class/power_supply/battery/current_now" '
        out  = subprocess.check_output(command).decode('utf-8')
        self.current = int(out)
        if self.current < 0:
            self.bool_power_supply = 0
        else:
            self.bool_power_supply = 1
        self.current = abs(self.current)/1000
        self.current_data.append(self.current)
  
        return self.current
    def getPower(self):
    # self.engine.enableChannel(sampleEngine.channels.MainCurrent)
    # self.engine.enableChannel(sampleEngine.channels.MainVoltage)
    # self.engine.startSampling(1)
    # sample = self.engine.getSamples()
    # current = sample[sampleEngine.channels.MainCurrent][0]
    # voltage = sample[sampleEngine.channels.MainVoltage][0]
    # self.Mon.stopSampling()
    # self.engine.disableChannel(sampleEngine.channels.MainCurrent)
    # self.engine.disableChannel(sampleEngine.channels.MainVoltage)
        self.power = self.getCurrent() * self.getVoltage()
       
        self.power_data.append(self.power)
      
        return self.power/1e3
    def print(self):
        print("Current mA:",self.getCurrent())
        print("Voltage mV:",self.getVoltage())
        print("Power",self.getPower())
if __name__ == "__main__":
    batter = PowerLogger(PHINE_IP,PHINE_PORT)
    print(batter.getPower())

