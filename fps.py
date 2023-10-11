import subprocess

app = ""

commad  =f'adb shell "dumpsys gfxinfo com.tencent.ig"'
out  = subprocess.check_output(commad)
print(out.split("---PROFILEDATA---")[1])