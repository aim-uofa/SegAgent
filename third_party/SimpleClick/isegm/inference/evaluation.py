from time import time

import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm
import pycocotools.mask as mask_util

def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious = []
    sample_path_list = dataset.dataset_samples
    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        # if index < len(dataset) - 100:
        #     continue
        if index >= predictor.start_index and index < predictor.end_index:
            pass 
        else:
            continue
        sample = dataset.get_sample(index)
        if sample is None:
            continue
        file_name = sample_path_list[index]
        if isinstance(file_name, dict):
            if 'coco_url' in file_name:
                file_name = '/'.join(file_name['coco_url'].split('/')[-2:])
                #file_name['coco_url'].split('/')[-2:]
        for object_id in sample.objects_ids:
            caption = sample._objects[object_id].get('caption', None)
            #print(caption)
            _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask(object_id), predictor,
                                                sample_id=index,  file_name=file_name, object_id=object_id, caption=caption, **kwargs)
            all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def convert_mask_to_coco_format(mask):
    # convert gt_mask to coco format
    mask = mask.astype(np.uint8)
    mask = mask_util.encode(np.array(mask, order='F'))
    if isinstance(mask, list):
        mask = mask_util.merge(mask)
    if isinstance(mask, dict):
        mask = {'counts': mask['counts'].decode('utf-8'), 'size': mask['size']}
    return mask

def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None, file_name=None,object_id=None, caption=None):
    clicker = Clicker(gt_mask=gt_mask)
    clicker.visualize = predictor.visualize
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []
    visualize = predictor.visualize
    record_trace = predictor.record_trace
    if record_trace:
        # record image name, height, width, gt_mask, clicks_list as coco format
        # one gt_mask 对应 一个 clicks_list , clicks_list 里面是每次点击的信息,mask 
        temp_dict = {}
        temp_dict['image_name'] = file_name
        # if 'person' in file_name:
        #     print(file_name)
        #     import torchshow
        #     torchshow.save(gt_mask>0)
        temp_dict['height'] = gt_mask.shape[0]
        temp_dict['width'] = gt_mask.shape[1]
        temp_dict['gt_ann'] = {}
        temp_dict['clicks_list'] = []
        # temp_dict['masks_list'] = []
        # temp_dict['ious_list'] = []
        temp_dict['gt_ann']['object_id'] = object_id
        temp_dict['gt_ann']['caption'] = caption
        gt_mask_coco = convert_mask_to_coco_format(gt_mask>0)
        temp_dict['gt_ann']['segmentation'] = gt_mask_coco
        temp_dict['gt_ann']['bbox'] = mask_util.toBbox(gt_mask_coco).tolist()
        temp_dict['gt_ann']['area'] = mask_util.area(gt_mask_coco).item()
    if isinstance(file_name, str):
        file_name = file_name.split('.')[0]
    with torch.no_grad():
        predictor.set_input_image(image)
        max_iou = 0
        for click_indx in range(max_clicks):
            clicker.object_id = object_id
            clicker.make_next_click(pred_mask,file_name) 
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr
           

            if visualize:
                visualize_dir = './visualizations'
                import os 
                if not os.path.exists(visualize_dir):
                    os.makedirs(visualize_dir)
                import cv2
                cv2.imwrite(f'{visualize_dir}/{file_name}_{str(object_id)}_pred_mask_{click_indx}.png', pred_mask.astype(np.uint8) * 255)
                # overlay mask on image
                overlay = image.copy()
                color = (0, 255, 0)  # Green color
                alpha = 0.5  # Transparency factor

                # Create a binary mask
                binary_mask = pred_mask.astype(np.uint8)

                # Create a color mask
                color_mask = np.zeros_like(image)
                color_mask[binary_mask > 0] = color

                # Apply the color mask on the original image
                for c in range(0, 3):
                    overlay[:, :, c] = np.where(binary_mask > 0,
                                                overlay[:, :, c] * (1 - alpha) + color_mask[:, :, c] * alpha,
                                                overlay[:, :, c])
                overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{visualize_dir}/{file_name}_{str(object_id)}_overlay_{click_indx}.png', overlay)
                

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_iou(gt_mask, pred_mask)
            #print(file_name, click_indx, iou)
            ious_list.append(iou)
            if record_trace:
                if iou > max_iou:
                    pred_mask_coco = convert_mask_to_coco_format(pred_mask)
                    temp_click = {}
                    temp_click['idx'] = click_indx
                    temp_click['coor'] = clicker.clicks_list[-1].coords
                    temp_click['is_positive'] = clicker.clicks_list[-1].is_positive
                    temp_click['mask'] = pred_mask_coco
                    temp_click['box'] = mask_util.toBbox(pred_mask_coco).tolist()
                    temp_click['area'] = mask_util.area(pred_mask_coco).item()
                    temp_click['iou'] = iou
                    temp_dict['clicks_list'].append(temp_click)
                else:
                    break
            if not record_trace:
                if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                    break
            else:
                # use different termination condition
                # 1.正常情况 
                if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                    #print(f'S1.file_name: {file_name}, object_id: {object_id}, iou: {iou}')
                    break
                # 2. iou 不再增加
                if iou > max_iou:
                    max_iou = iou
                else:
                    #print(f'S2.file_name: {file_name}, object_id: {object_id}, iou: {iou}')
                    break
        print(f'file_name: {file_name}, object_id: {object_id}, iou: {iou}')
        if visualize:
                cv2.imwrite(f'{visualize_dir}/{file_name}_{str(object_id)}_gt_mask.png', gt_mask.astype(np.uint8) * 255)
                # overlay mask on image
                overlay = image.copy()
                color = (0, 255, 0)  # Green color
                alpha = 0.5
                # Create a binary mask
                binary_mask = gt_mask # gt_mask may < 0
                # Create a color mask
                color_mask = np.zeros_like(image)
                color_mask[binary_mask > 0] = color
                # Apply the color mask on the original image
                for c in range(0, 3):
                    overlay[:, :, c] = np.where(binary_mask > 0,
                                                overlay[:, :, c] * (1 - alpha) + color_mask[:, :, c] * alpha,
                                                overlay[:, :, c])
                overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{visualize_dir}/{file_name}_{str(object_id)}_overlay_gt.png', overlay)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{visualize_dir}/{file_name}_{str(object_id)}_image.png', image)
        show_video = False
        if show_video:
            import cv2
            import os
            import glob
            # Define the codec and create VideoWriter object
            out = cv2.VideoWriter(f'{visualize_dir}_videos/{file_name}_overlay.avi', cv2.VideoWriter_fourcc(*'DIVX'), 1, (image.shape[1], image.shape[0]))
            if not os.path.exists(f'{visualize_dir}_videos'):
                os.makedirs(f'{visualize_dir}_videos')
            for filename in glob.glob(f'{visualize_dir}/{file_name}_{str(object_id)}_overlay_*.png'):
                img = cv2.imread(filename)
                out.write(img)
            out.release()
            # other video , visualizations/153093_next_click_position_1.png
            out = cv2.VideoWriter(f'{visualize_dir}_videos/{file_name}_{str(object_id)}_clicks.avi', cv2.VideoWriter_fourcc(*'DIVX'), 1, (image.shape[1], image.shape[0]))
            for filename in glob.glob(f'{visualize_dir}/{file_name}_{str(object_id)}_next_click_position_*.png'):
                img = cv2.imread(filename)
                out.write(img)
            out.release()
        
        if record_trace:
            predictor.record_json['data'].append(temp_dict)
        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
