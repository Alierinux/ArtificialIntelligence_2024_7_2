1、确认是否有英伟达显卡
2、终端执行命令：nvidia-smi
3、下载安装英伟达驱动（注意版本型号）
4、终端执行命令：nvidia-smi  查看：CUDA Version: 12.5
5、安装pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
