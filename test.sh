#!/bin/bash

# 设置循环次数
NUM_RUNS=100  # 你可以根据需要调整循环次数

# 设置每次运行后的等待时间（秒）
WAIT_TIME=30  # 如果需要等待，可以设置一个时间，否则设置为0

# 循环运行命令
for ((i=1; i<=NUM_RUNS; i++))
do
    echo "Running iteration $i of $NUM_RUNS..."

    # 运行你的命令
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py -y /home/yj/HEAL/opencood/hypes_yaml/opv2v/LiDAROnly/lidar_pyramid_revised.yaml

    # 检查命令是否成功
    if [ $? -eq 0 ]; then
        echo "Iteration $i completed successfully."
    else
        echo "Iteration $i failed."
        break  # 如果命令失败，退出循环
    fi

    # 如果不是最后一次运行，等待一段时间
    if [ $i -lt $NUM_RUNS ]; then
        echo "Waiting for $WAIT_TIME seconds before the next run..."
        sleep $WAIT_TIME
    fi
done

echo "All runs completed."