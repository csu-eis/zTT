import time
import argparse
import subprocess
import re

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from setting import *
nanoseconds_per_second = 1e9

class SurfaceFlingerFPS():

	def __init__(self,ip,port,keyword):
		
		self.ip = f"{ip}:{port}"
		self.ip = "10105252CF000197"
		self.fps = 0
		self.hardware_fps = 0
		self.view = self.get_sufaceView(keyword)
  
	def get_sufaceView(self,keyword):
		commad = f'adb  -s {self.ip} shell "dumpsys SurfaceFlinger --list"'
		out  = subprocess.check_output(commad).decode('utf-8')
		name = ""
		for i in out.split('\n'):
			i.strip()
			if i.startswith("SurfaceView") and "BLAST" in i and keyword in i:
				print(i)
				name = i.replace("\r", "")
				self.get_sufaceView = name
				break
		print("name",name)
		return name
		
	def get_fps(self):
   
		commad = f'adb  -s {self.ip} shell     "dumpsys SurfaceFlinger --latency  {repr(self.view)}  "    '
		out  = subprocess.check_output(commad).decode('utf-8')
		hardware = 0
		fps_list = [ line.replace("\r","").split('\t')  for line in out.split('\n') ]
		# print(fps_list)
		hardware = 1000/(int(fps_list[0][0])/1e6)
		fps = 127/(int(fps_list[-3][0]) - int(fps_list[1][0]) )*1e9
		self.hardware_fps = hardware
		self.fps = fps
  
		# print(hardware,fps)
		return fps	

	def check_top_app(self):
		""" 检查top app 是否是 目标app"""
		
	

if __name__ == "__main__":
    app ="org.videolan.vlc"
    fps = SurfaceFlingerFPS(PHINE_IP,PHINE_PORT,keyword=app)
    
    fps.get_fps()
    print(fps.fps)
    