#!/bin/bash

# 该脚本是模型训练的入口脚本。

# 打印当前脚本所在的目录，用于调试和确认环境。
# ${RUNTIME_SCRIPT_DIR} 是一个环境变量，通常由执行环境（如竞赛平台）提供，指向脚本所在的目录。
echo "Current script directory: ${RUNTIME_SCRIPT_DIR}"

# 进入训练工作区。
# 为了确保脚本中的相对路径能够正确解析（例如，加载数据或模型），
# 通常需要将当前目录切换到脚本所在的目录。
cd ${RUNTIME_SCRIPT_DIR}

# 执行训练命令。
# `python -u main.py`
# - `python`: 启动 Python 解释器。
# - `-u`: 参数表示不使用缓冲，这样可以确保日志（如print语句的输出）能够实时地被重定向或显示，而不会因为缓冲而延迟。
# - `main.py`: 这是训练程序的主文件。
echo "Starting training..."
python -u main.py
echo "Training finished."
