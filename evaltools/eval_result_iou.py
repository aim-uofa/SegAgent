 
"""
Evaluation script for computing IoU (Intersection over Union) metrics 
on segmentation results across RefCOCO family datasets.

This script processes prediction JSON files and computes IoU metrics
comparing predicted masks with ground truth masks.
"""

import os
import argparse
import csv
import json
import numpy as np
from tqdm import tqdm
import sys
import torch
import torch.distributed as dist
from pycocotools import mask as maskUtils

# Add parent directory to path for utility imports
sys.path.append('../../')
from utils import intersectionAndUnionGPU, AverageMeter, Summary
def forward_single(data, click_id=None):
    """
    Extract prediction mask from data based on click_id.
    
    Args:
        data (dict): Data dictionary containing prediction list
        click_id (int, optional): Specific click ID to extract. If None, 
                                 finds the prediction with highest IoU.
    
    Returns:
        dict: Mask dictionary in COCO format
    """
    if click_id is None:
        pred_list = data['pred_list']
        mask_list = []
        
        for sample in pred_list:
            mask = sample['mask']
            mask_list.append(mask)
        
        # Extract IoU values from outputs
        try:
            iou_list = [
                float(sample['outputs'].split('Current IOU: ')[-1].split(',')[0]) 
                if sample['outputs'] is not None else 0 
                for sample in pred_list
            ]
        except:
            try:
                iou_list = [
                    float(sample['outputs'].split('Current IOU: ')[-1].split('P')[0].split('N')[0]) 
                    if sample['outputs'] is not None else 0 
                    for sample in pred_list
                ]
            except:
                iou_list = [0] * len(pred_list)
        
        if len(iou_list) == 0:
            return mask_list[0] if len(mask_list) > 0 else None
        
        max_iou_idx = np.argmax(iou_list)
        return mask_list[max_iou_idx-1] if max_iou_idx > 0 else mask_list[0]
    
    return data['pred_list'][click_id]['mask']


def annToMask(mask_ann, h, w):
    """
    Convert annotation to mask array.
    
    Args:
        mask_ann: Mask annotation in various formats
        h (int): Height of the mask
        w (int): Width of the mask
    
    Returns:
        np.ndarray: Binary mask array
    """
    if mask_ann is None:
        return np.zeros((h, w), dtype=np.uint8)

    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def initialize_distributed():
    """Initialize distributed training environment."""
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(np.random.randint(10000, 20000))
    
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')


def evaluate_dataset(input_json, click_id=None, visualize=False):
    """
    Evaluate a single dataset and compute IoU metrics.
    
    Args:
        input_json (str): Path to input JSON file
        click_id (int, optional): Specific click ID to evaluate
        visualize (bool): Whether to save visualizations
    
    Returns:
        tuple: (class_iou, global_iou) metrics
    """
    with open(input_json, 'r') as f:
        data_ = json.load(f)

    output_csv = input_json.replace('.json', f'_clicknew2{click_id}.csv')
    
    if os.path.exists(output_csv):
        print(f'{output_csv} already exists, skipping...')
        return None, None
    
    trackers = {
        "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
        "union": AverageMeter("Union", ":6.3f", Summary.SUM),
        "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
    }
    
    workspace = os.path.dirname(input_json)
    
    with open(output_csv, 'w') as fw:
        writer = csv.writer(fw)
        writer.writerow(['image_name', 'mask_idx', 'iou', 'caption'])
        
        previous_image_name = ''
        pred_mask_list = []
        gt_mask_list = []
        caption_list = []
        id = 0
        
        for i, data in tqdm(enumerate(data_), desc=f"Processing {os.path.basename(input_json)}"):
            if data['img_path'] != previous_image_name:
                # Process accumulated data for previous image
                if previous_image_name != '' and len(gt_mask_list) > 0:
                    _process_image_masks(gt_mask_list, pred_mask_list, caption_list, 
                                       previous_image_name, writer, trackers)
                
                # Reset for new image
                pred_mask_list = []
                gt_mask_list = []
                caption_list = []
                previous_image_name = data['img_path']

            # Process current data sample
            data['annotation'] = [{'segmentation': data['masks'], 'id': id, 'click_id': data['click_id']}]
            
            pred_mask = forward_single(data, click_id)
            if pred_mask is None:
                continue
                
            pred_mask = annToMask(pred_mask, data['height'], data['width'])
            pred_mask = torch.from_numpy(pred_mask).unsqueeze(0)

            gt_mask = annToMask(data['gt_mask'], data['height'], data['width'])
            gt_mask = torch.from_numpy(gt_mask).unsqueeze(0)
            
            gt_mask_list.append(gt_mask)
            pred_mask_list.append(pred_mask)
            caption_list.append(data['caption'][0])
            id += 1

        # Process final image
        if len(gt_mask_list) > 0:
            _process_image_masks(gt_mask_list, pred_mask_list, caption_list, 
                               previous_image_name, writer, trackers)

    # Calculate final metrics
    for meter in trackers.values():
        meter.all_reduce()
    
    iou_per_class = trackers["intersection"].sum / (trackers["union"].sum + 1e-5)
    class_iou = iou_per_class[1]
    global_iou = trackers["gIoU"].avg[1]
    
    # Save results
    txt_path = output_csv.replace('.csv', f'_click{click_id}.txt')
    with open(txt_path, 'w') as f:
        f.write(f'class_iou: {class_iou}\n')
        f.write(f'global_iou: {global_iou}\n')
    
    print(f'Results saved to: {output_csv}')
    print(f'Class IoU: {class_iou:.4f}')
    print(f'Global IoU: {global_iou:.4f}')
    
    return class_iou, global_iou


def _process_image_masks(gt_mask_list, pred_mask_list, caption_list, 
                        image_name, writer, trackers):
    """Helper function to process masks for a single image."""
    gt_mask_list = torch.stack(gt_mask_list).squeeze(1)
    pred_mask_list = torch.stack(pred_mask_list).squeeze(1)

    intersection, union, accuracy_iou = 0.0, 0.0, 0.0
    
    for target, prediction, caption_, mask_idx in zip(
        gt_mask_list, pred_mask_list, caption_list, range(len(gt_mask_list))
    ):
        intersect, union_, _ = intersectionAndUnionGPU(
            prediction.contiguous().clone().float(), 
            target.contiguous().float(), 
            2, ignore_index=255
        )
        intersection += intersect
        union += union_
        accuracy_iou += intersect / (union_ + 1e-5)
        accuracy_iou[union_ == 0] += 1.0
        
        iou_value = (intersect / (union_ + 1e-5))[-1].item()
        writer.writerow([image_name.split('/')[-1], mask_idx, iou_value, caption_])
    
    intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
    accuracy_iou = accuracy_iou.cpu().numpy() / len(gt_mask_list)
    
    writer.writerow([image_name.split('/')[-1], -1, accuracy_iou[-1].item(), ''])
    trackers["intersection"].update(intersection)
    trackers["union"].update(union)
    trackers["gIoU"].update(accuracy_iou, n=len(gt_mask_list))
def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate IoU metrics for segmentation results')
    parser.add_argument('--input_json', type=str, required=True,
                       help='Path to input JSON file with predictions')
    parser.add_argument('--cuda_device', type=str, default='0',
                       help='CUDA device to use (default: 0)')
    parser.add_argument('--click_id', type=int, default=None,
                       help='Specific click ID to evaluate (default: use best IoU)')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization results')
    parser.add_argument('--eval_all_splits', action='store_true',
                       help='Evaluate all RefCOCO family splits')
    
    args = parser.parse_args()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    # Initialize distributed environment
    initialize_distributed()
    
    if args.eval_all_splits:
        # Evaluate all standard splits
        desired_order = [
            ('refcocog', 'umd', 'val'),
            ('refcoco', 'unc', 'val'),
            ('refcoco', 'unc', 'testA'),
            ('refcoco', 'unc', 'testB'),
            ('refcoco+', 'unc', 'val'),
            ('refcoco+', 'unc', 'testA'),
            ('refcoco+', 'unc', 'testB'),
            ('refcocog', 'umd', 'test')
        ]
        
        results = {}
        for split in desired_order:
            dataset, split_by, split_name = split
            print(f"\n=== Evaluating {dataset}_{split_name} ===")
            
            input_json = args.input_json.replace('refcoco+_val', f'{dataset}_{split_name}')
            
            if not os.path.exists(input_json):
                print(f'{input_json} not found, skipping...')
                continue
            
            class_iou, global_iou = evaluate_dataset(input_json, args.click_id, args.visualize)
            if class_iou is not None:
                results[f'{dataset}_{split_name}'] = {
                    'class_iou': float(class_iou),
                    'global_iou': float(global_iou)
                }
        
        # Print summary
        print("\n=== Evaluation Summary ===")
        for split_name, metrics in results.items():
            print(f"{split_name}: Class IoU = {metrics['class_iou']:.4f}, "
                  f"Global IoU = {metrics['global_iou']:.4f}")
            
    else:
        # Evaluate single file
        if not os.path.exists(args.input_json):
            raise FileNotFoundError(f"Input file not found: {args.input_json}")
        
        evaluate_dataset(args.input_json, args.click_id, args.visualize)


if __name__ == "__main__":
    main()

                        