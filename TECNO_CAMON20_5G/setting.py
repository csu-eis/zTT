"""IP Address Setting"""
PHINE_IP = "172.16.101.79"
PHINE_PORT = 5555

SERVER_IP="127.0.0.1"
SERVER_PORT=8702



"""Targe Setting"""
TARGET_FPS = [10,13,16,19,22]
TARGET_TEMP = 70
#TARGET_APP = "com.tencent.ig"
# TARGET_APP = "org.videolan.vlc"
TARGET_APP = "mobi.eis.cocpuandgpu.MultiPersonsEstimationActivity"

"""Train Setting"""
EXPERIMENT_TIME=5000000
BUFFER_SIZE = 1600000

GAMMA = 0.5
ALPHA = 0.5
SAVE_PATH = f"./save_model/{'_'.join(str(num) for num in sorted(TARGET_FPS))}"  # './save_model/9_10_20_30'
IS_TRAIN = False
BATCH_SIZE = 16

ARRIVED_TIME = 10
ACTION_SPACE = 3 # CPU0 CPU1 CPU2
BRANCHES = [3,3,3]  #0-16 0-16 0-16
S_DIM=4  #状态信息 4个 CPU0-3 clock fps - target
H_DIM=32  #隐藏层大小


# clock_change_time=30
# cpu_power_limit=1000
# gpu_power_limit=1600

"""Other"""
CSV_PATH = ""