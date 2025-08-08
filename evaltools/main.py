from config import get_config
import torch
import os
import sys
sys.path.append('../third_party/SimpleClick')
import torch.distributed as dist
from model_loader import load_model
from refcocog_eval import REFCOCOG_EVAL

def initialize_distributed():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    # 将 MASTER_PORT 设置为 12345 + CUDA_VISIBLE_DEVICES 的值
    os.environ["MASTER_PORT"] = str(12399 + int(cuda_visible_devices))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

if __name__ == "__main__":
    args = get_config()
    # SimpleClick
    if args.cpu:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.gpus.split(',')[0]}")

    if (args.iou_analysis or args.print_ious) and args.min_n_clicks <= 1:
        args.target_iou = 1.01
    else:
        args.target_iou = max(0.8, args.target_iou)
    # import debugpy
    # debugpy.listen(('localhost', 5678))
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()    
    initialize_distributed()

    segmentation_model, grounding_model = load_model(args)

    refcocog_eval = REFCOCOG_EVAL(grounding_model, segmentation_model, args)
    # refcocog_eval.predictor = predictor
    refcocog_eval.forward(args.img, args.json)
