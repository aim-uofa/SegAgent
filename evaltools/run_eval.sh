#!/bin/bash

# 目标目录
TARGET_DIR="/home/zmz/code/Osprey-main/evaltools"

# 切换到脚本目录
if [ "$(pwd)" != "$TARGET_DIR" ]; then
  cd "$TARGET_DIR" || exit
fi

# 获取模型路径作为第一个参数
MODEL_PATH=$1
if [ -z "$MODEL_PATH" ]; then
  echo "请提供模型路径作为参数：bash run_eval.sh /path/to/your/model"
  exit 1
fi

# 固定配置参数
CHECKPOINT_PATH="/home/zmz/code/SimpleClick/weights/cocolvis_vit_large.pth"
DATA_PATH="/home/zmz/code/Osprey-main/data"

# 设定点击次数（可改为范围如 1..5）
for n_clicks in {7..7}; do
    export VIS_DIR="$MODEL_PATH"

    CUDA_VISIBLE_DEVICES=4 python main.py NoBRS \
      --model "$MODEL_PATH" \
      --img "$DATA_PATH/refer_seg/images" \
      --json "$DATA_PATH/segaction/refcoco_testA.json" \
      --checkpoint "$CHECKPOINT_PATH" \
      --no_use_mask_module \
      --use_previous_mask 1 \
      --n-clicks "$n_clicks" \
      --seg_model simple_click \
      --grounding_model qwen-full
done
