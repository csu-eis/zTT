# zTT 基于DVFS调节
# 使用
1. 首先打开pubg 保证在pubg下运行代码
2. 在setting.py 下设置adb ip port
3. 运行agent.py
4. 在clien中设置迭代次数 以及目标fps 运行client.py 
# todo
- [x] 调试移植
- [ ] 微调
  
# 说明
原项目链接 [地址](https://github.com/ztt-21/zTT)  
论文地址[地址](https://dl.acm.org/doi/10.1145/3458864.3468161)

# 一些指令
```bash
cd /sys/module/ged/parameters/gx_top_app_pid
cat gx_top_app_pid
top | grep 
shell dumpsys activity top
```