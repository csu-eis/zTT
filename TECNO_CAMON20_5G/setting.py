"""IP Address Setting"""
PHINE_IP = "172.16.101.79"
PHINE_PORT = 42307

SERVER_IP="127.0.0.1"
SERVER_PORT=8702

"""Train Setting"""
EXPERIMENT_TIME=5000
ACTION_SPACE = 16*40
CLOCK_CHANGE__TIME = 30
CLOCK_CHANGE__LIMIT = 1000
GPU_POWER_LIMIT=1200
PICKLE_PATH = "./save_model/agent.data"
IS_TRAIN = False
# clock_change_time=30
# cpu_power_limit=1000
# gpu_power_limit=1600

"""Targe Setting"""
TARGET_FPS = 18
TARGET_TEMP = 70
#TARGET_APP = "com.tencent.ig"
# TARGET_APP = "org.videolan.vlc"
TARGET_APP = "mobi.eis.cocpuandgpu.MultiPersonsEstimationActivity"
"""Other"""
CSV_PATH = f"./{TARGET_APP}_{TARGET_FPS}_{TARGET_TEMP}_{EXPERIMENT_TIME}.csv"