# load config.yml
config_path = './config.yml'
json_path = '/home/zmz/code/SimpleClick/data/record_trace/GrabCut.json'
json_path = 'data/record_trace/PascalVOC.json'
json_path = 'data/record_trace/ReasonSeg_train.json'
json_path = 'data/record_trace/refclef_val.json'
# data/record_trace/refcoco_val.json
# data/record_trace/refcocog_val.json
# data/record_trace/refcoco+_val.json
json_path = 'data/record_trace/refcoco_val.json'
#json_path = 'data/record_trace/refcocog_val.json'
#json_path = 'data/record_trace/refcoco+_val.json'
json_path = 'data/record_trace/LVIS_v1_val.json'
import yaml, json, os,torch
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
with open(json_path, 'r') as f:
    record = json.load(f)
dataset_name = json_path.split('/')[-1].split('.')[0]
split = 'train' if 'train' in dataset_name else 'val'
dataset_name = dataset_name.split('_')[0]
DATA_PATH_MAP = {
    'GrabCut': 'data/GrabCut/data_GT',
    'PascalVOC': 'data/VOCdevkit/VOC2012/JPEGImages',
    'ReasonSeg': 'dataset/reason_seg/ReasonSeg',
    'refclef': 'dataset/refer_seg/images/saiapr_tc-12',
    'refcoco': 'dataset/refer_seg/images/mscoco/images/train2014',
    'refcoco+': 'dataset/refer_seg/images/mscoco/images/train2014',
    'refcocog': 'dataset/refer_seg/images/mscoco/images/train2014',
    'LVIS': 'dataset/coco',
}
data_path = DATA_PATH_MAP[dataset_name]
if dataset_name == 'ReasonSeg':
    data_path += f'/{split}'
# visualize img / mask / points / gt_mask
import cv2
from pycocotools import mask as mask_util
import numpy as np
import torchshow
visualize_dir = '/home/zmz/code/SimpleClick/data/record_trace/vis/'
visualize_dir += dataset_name + '_' + split
if not os.path.exists(visualize_dir):
    os.makedirs(visualize_dir)

# def get_image_path(self, image_id):
#         if self.DatasetType == "refcoco" or self.DatasetType == "refcoco+" or self.DatasetType == "refcocog":
#             return self.dataset_path + "refer_seg/images/mscoco/images/train2014/COCO_train2014_000000" + str(image_id).zfill(6) + ".jpg"
#         elif self.DatasetType == "refclef":
#             return self.dataset_path + "refer_seg/images/saiapr_tc-12/" + str(image_id // 1000).zfill(2) + "/images/" + str(image_id) + ".jpg"
#         else:
#             raise RuntimeError("Error: dataset not found")
def get_image_path(image_id, dataset_name):
    if dataset_name == 'refclef':
        return f'{str(image_id // 1000).zfill(2)}/images/{str(image_id)}.jpg'
    elif dataset_name == 'refcoco' or dataset_name == 'refcoco+' or dataset_name == 'refcocog':
        return f'COCO_train2014_000000{str(image_id).zfill(6)}.jpg'
def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    overlay = image.copy()
    binary_mask = mask
    color_mask = np.zeros_like(image)
    color_mask[binary_mask > 0] = color
    for c in range(0, 3):
        overlay[:, :, c] = np.where(binary_mask > 0,
                                    overlay[:, :, c] * (1 - alpha) + color_mask[:, :, c] * alpha,
                                    overlay[:, :, c])
    return overlay

def overlay_points(image, points, color=(255, 0, 0)):
    overlay = image.copy()
    cv2.circle(overlay, (points[1],points[0]), 5, color, -1)
    return overlay

def visualize(image,masks,points,path):
    # check image type
    if isinstance(image, torch.Tensor):
        image = image.permute(1,2,0).cpu().numpy() # C,H,W -> H,W,C
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if points[0] < 1:
        points = points * image.shape[:2]
    # overlay mask on image
    overlay = overlay_mask(image, masks)
    overlay = overlay_points(overlay, points)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    cv2.imwrite(path, overlay)
    return overlay

def visualize(record, data_path, visualize_dir,idx,dataset_name):
    data = record['data'][idx]
    image_name = data['image_name']
    if dataset_name in ['refclef', 'refcoco', 'refcoco+', 'refcocog']:
        image_path = os.path.join(data_path, get_image_path(image_name, dataset_name))
    else:
        if '.' not in image_name:
            image_name += '.jpg'
        image_path = os.path.join(data_path, image_name)
    image = cv2.imread(image_path)
    gt_mask = mask_util.decode(data['gt_ann']['segmentation'])
    object_id = data['gt_ann']['object_id'] # one image may have multiple objects
    if '/' in image_name:
        image_name = image_name.split('/')[-1]
    # if 'person' in image_name:
    #     print(image_name)
    #     torchshow.save(gt_mask)
    overlay_image = overlay_mask(image, gt_mask)
    gt_mask_area = gt_mask.sum()
    cv2.imwrite(f'{visualize_dir}/{image_name}_{object_id}_gt_mask_{gt_mask_area}.png', gt_mask.astype(np.uint8) * 255)
    cv2.imwrite(f'{visualize_dir}/{image_name}_{object_id}_overlay_gt.png', overlay_image)
    caption = data['gt_ann']['caption']
    # recort text
    with open(f'{visualize_dir}/{image_name}_{object_id}_gt_caption.txt', 'w') as f:
        if isinstance(caption, list):
            for i, cap in enumerate(caption):
                f.write(f'{i}: {cap}\n')
        else:
            f.write(caption)
    # visualize click_lists
    clicks_list = data['clicks_list']
    for i, clicks in enumerate(clicks_list):
        temp_mask = mask_util.decode(clicks['mask'])
        temp_point = clicks['coor']
        temp_iou = clicks['iou']
        temp_iou = round((round(temp_iou, 4) * 100),4)
        overlay_image = overlay_mask(image, temp_mask)
        if clicks['is_positive']:
            overlay_image = overlay_points(overlay_image, temp_point, color=(255, 0, 0))
        else:
            overlay_image = overlay_points(overlay_image, temp_point, color=(0, 0, 255))
        #overlay_image = overlay_points(overlay_image, temp_point)
        cv2.imwrite(f'{visualize_dir}/{image_name}_{object_id}_click_{i}_{temp_iou}.png', overlay_image)
    




for i in range(len(record['data'])):
    if i%10 != 0 or i < 100:
        continue
    visualize(record, data_path, visualize_dir, i,dataset_name)

print('Visualizations saved in', visualize_dir)



    