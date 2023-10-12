import time
import argparse
import subprocess
import re

import sys
sys.path.append(".") 
from setting import *
nanoseconds_per_second = 1e9

class SurfaceFlingerFPS():

	def __init__(self, view, ip,port):
		self.view = view
		self.ip = f"{ip}:{port}"
	
		self.fps = 0
		self.hardware_fps = 0
	def get_sufaceView(self):
		commad = f'adb  -s {self.ip} shell "dumpsys SurfaceFlinger --list"'
		out  = subprocess.check_output(commad).decode('utf-8')
		for i in out.split('\n'):
			i.strip()
			if i.startswith("SurfaceView") and "BLAST" in i:
				name = i.replace("\r", "")
				self.get_sufaceView = name
		return name
		
	def get_fps_list(self):
   
		commad = f'adb  -s {self.ip} shell     "dumpsys SurfaceFlinger --latency  {repr(self.get_sufaceView())}  "    '
		out  = subprocess.check_output(commad).decode('utf-8')
		hardware = 0
		fps_list = [   line.replace("\r","").split('\t')           for line in out.split('\n') ]
		# print(fps_list)
		hardware = 1000/(int(fps_list[0][0])/1e6)
		fps = 127/(int(fps_list[-3][0]) - int(fps_list[1][0]) )*1e9
		self.hardware_fps = hardware
		self.fps = fps
  
		print(hardware,fps)
		return fps	
		
	

if __name__ == "__main__":
    view ="com.android.chrome"
    fps = SurfaceFlingerFPS(view,IP,PORT)
 
    fps.get_fps_list()
    