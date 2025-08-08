import numpy as np
import torch
import json
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import pycocotools.mask as maskUtils
import sys
sys.path.append('/mnt/nas/share/home/zmz/code/SimpleClick/')
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from tqdm import tqdm
#dataset_path = "/mnt/nas/share/home/zmz/code/SimpleClick/dataset/"


class ReferSegDataset(ISDataset):
    def __init__(self, dataset_path,dataset_type, split, **kwargs):
        super(ReferSegDataset, self).__init__(**kwargs)
        
        # Load ref file
        if dataset_type == "refcoco" or dataset_type == "refcoco+" or dataset_type == "refclef":
            ref_file = os.path.join(dataset_path + "refer_seg/" + dataset_type + "/refs(unc).p")
        elif dataset_type == "refcocog":
            ref_file = os.path.join(dataset_path + "refer_seg/" + dataset_type + "/refs(google).p")
        refs = pickle.load(open(ref_file, "rb"))
        splited_refs = [ref for ref in refs if ref['split'] == split]
        self.dataset_path = dataset_path
        self.Split = split
        # Load inst file
        with open(dataset_path + "refer_seg/" + dataset_type + "/instances.json", "r") as file:
            inst = json.load(file)
        
        # Construct the reflection from idx to imgs
        idx2imgs = []
        for ref in splited_refs:
            image_id = ref['image_id']
            if image_id not in idx2imgs:
                idx2imgs.append(image_id)

        # Construct the reflection from img to refs
        img2refs = {}
        img_total = 0; ann_total = 0;
        for ref in splited_refs:
            image_id = ref['image_id']
            if image_id not in img2refs:
                img2refs[image_id] = []
                img_total += 1
            img2refs[image_id].append(ref)
            ann_total += 1
        print(f'[INFO] Totally {img_total} images and {ann_total} annotations in the dataset')

        # Construct the dictionary of images in instance.json
        imgs = {}
        for entry in inst['images']:
            imgs[entry['id']] = entry

        # Construct the dictionary of anns in instance.json
        anns = {}
        for entry in inst['annotations']:
            anns[entry['id']] = entry

        self.DatasetType = dataset_type
        self.ImgTotal = img_total
        self.AnnTotal = ann_total

        self.Idx2imgs = idx2imgs
        self.Img2refs = img2refs
        self.Refs = splited_refs
        self.Inst = inst
        self.Imgs = imgs
        self.Anns = anns

        self.dataset_samples = idx2imgs # [NOTE] Just to make the code compatible with super class
        self.missing_record = []

    def get_image_path(self, image_id):
        if self.DatasetType == "refcoco" or self.DatasetType == "refcoco+" or self.DatasetType == "refcocog":
            return self.dataset_path + "refer_seg/images/mscoco/images/train2014/COCO_train2014_000000" + str(image_id).zfill(6) + ".jpg"
        elif self.DatasetType == "refclef":
            return self.dataset_path + "refer_seg/images/saiapr_tc-12/" + str(image_id // 1000).zfill(2) + "/images/" + str(image_id) + ".jpg"
        else:
            raise RuntimeError("Error: dataset not found")

    def get_sample(self, idx) -> DSample:
        img_id = self.Idx2imgs[idx]
        refs = self.Img2refs[img_id]

        # Some images in ReferSeg cannot be found...?
        if not os.path.exists(self.get_image_path(img_id)):
            self.missing_record.append(self.get_image_path(img_id))
            return None
        image = cv2.imread(self.get_image_path(img_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_masks = []
        objects_ids = []
        ann_idx = 0
        instance_to_class = {}
        for ref in refs:
            ann_id = ref['ann_id']
            entry = self.Anns[ann_id]
            if self.DatasetType == "refclef":
                rle = entry['segmentation'][0]
                gt_mask = maskUtils.decode(rle)
                gt_mask = gt_mask.reshape(gt_mask.shape[0], gt_mask.shape[1])
            elif entry['iscrowd'] == 1:
                rle = maskUtils.frPyObjects(entry['segmentation'], entry['segmentation']['size'][0], entry['segmentation']['size'][1])
                gt_mask = maskUtils.decode(rle)
                gt_mask = gt_mask.reshape(gt_mask.shape[0], gt_mask.shape[1])
            else:
                gt_mask = np.zeros((self.Imgs[img_id]['height'], self.Imgs[img_id]['width']))
                polygens = entry['segmentation']
                for polygen in polygens:
                    polygen = np.array(polygen)
                    polygen = np.hstack((polygen[0::2].reshape(-1, 1), polygen[1::2].reshape(-1, 1)))
                    gt_mask = cv2.fillPoly(gt_mask, [polygen.astype(np.int32)], 1)
            
            gt_masks.append(gt_mask)
            objects_ids.append((ann_idx, 1))
            instance_to_class[(ann_idx, 1)] = [ ref['sentences'][i]['sent'] for i in range(len(ref['sentences'])) ]
            if len(ref['sentences']) > 1:
                print(f'[INFO] Multiple captions found in {idx}')
            ann_idx += 1

        gt_masks = np.array(gt_masks).transpose(1, 2, 0)
        if image.shape[0] != gt_masks.shape[0] or image.shape[1] != gt_masks.shape[1]:
            if image.shape[0] == gt_masks.shape[1] and image.shape[1] == gt_masks.shape[0]:
                image = np.flip(image.transpose(1, 0, 2), axis=1)
                print(f'[INFO] Transpose image found in {self.get_image_path(img_id)}')
                return DSample(image, gt_masks, objects_ids=objects_ids, sample_id=idx, instance_to_class=instance_to_class)
            else:
                print(f'[INFO] Shape mismatch found in {self.get_image_path(img_id)}')
                return None
        return DSample(image, gt_masks, objects_ids=objects_ids, sample_id=idx, instance_to_class=instance_to_class)
    
    def save_missing_record(self):
        if len(self.missing_record) == 0:
            print(f'[INFO] No missing record found')
            return
        save_path = self.dataset_path + "refer_seg/" + self.DatasetType + f"/missing_record_{self.Split}.txt"
        with open(save_path, "w") as file:
            for record in self.missing_record:
                file.write(record + "\n")
        print(f'[INFO] Missing record saved to {save_path}')
        

class ReasonSegDataset(ISDataset):
    def __init__(self, dataset_path,split, **kwargs):
        super(ReasonSegDataset, self).__init__(**kwargs)

        sample_set = []
        for root, dirs, files in os.walk(dataset_path + "reason_seg/ReasonSeg/" + split):
            for file in files:
                if file.endswith(".jpg"):
                    file_path = os.path.join(root, file)
                    sample_set.append(file[:-4])

        self.ImgTotal = len(sample_set)
        self.split = split
        print(f'[INFO] Totally {self.ImgTotal} annotations in the dataset')
        self.dataset_path = dataset_path
        self.dataset_samples = sample_set
        self.missing_record = []

    def get_sample(self, idx) -> DSample:
        with open(self.dataset_path + "reason_seg/ReasonSeg/"+ self.split + "/" + self.dataset_samples[idx] + ".json", "r") as file:
            content = json.load(file)

        image = cv2.imread(self.dataset_path + "reason_seg/ReasonSeg/" + self.split + "/" + self.dataset_samples[idx] + ".jpg")

        # Some shapes in ReasonSeg is empty...?
        if len(content['shapes']) == 0:
            self.missing_record.append(self.dataset_path + "reason_seg/ReasonSeg/" + self.split + "/" + self.dataset_samples[idx] + ".jpg")
            return None
        gt_mask = np.zeros((image.shape[0], image.shape[1]))
        for shape in content['shapes']:
            polygon = np.array(shape['points'])
            gt_mask = cv2.fillPoly(gt_mask, [polygon.astype(np.int32)], 1)
        isinstance_to_class = {(0, 1): content['text'] }
        if content['is_sentence'] != True:
            print(f'[INFO] Not sentence found in {idx}')
            
        return DSample(image, gt_mask, objects_ids=[1], sample_id=idx, instance_to_class=isinstance_to_class)
    
    def save_missing_record(self):
        if len(self.missing_record) == 0:
            print(f'[INFO] No missing record found')
            return
        save_path = self.dataset_path + "reason_seg/ReasonSeg/"+f"missing_record_{self.split}.txt"
        with open(save_path, "w") as file:
            for record in self.missing_record:
                file.write(record + "\n")
        print(f'[INFO] Missing record saved to {save_path}')
        
if __name__ == '__main__':
    # check dataset
    # reasonseg = ReasonSegDataset("/mnt/nas/share/home/zmz/code/SimpleClick/dataset/", "train")
    # for i in tqdm(range(reasonseg.ImgTotal)):
    #     sample = reasonseg.get_sample(i)
    #     if sample is None:
    #         print(f'[INFO] Empty annotation found in {i}')
    # reasonseg.save_missing_record()
    # print(f'[INFO] ReasonSeg train dataset check done')
    # reasonseg = ReasonSegDataset("/mnt/nas/share/home/zmz/code/SimpleClick/dataset/", "val")
    # for i in tqdm(range(reasonseg.ImgTotal)):
    #     sample = reasonseg.get_sample(i)
    #     if sample is None:
    #         print(f'[INFO] Empty annotation found in {i}')
    # reasonseg.save_missing_record()
    # print(f'[INFO] ReasonSeg val dataset check done')
    # referseg = ReferSegDataset("/mnt/nas/share/home/zmz/code/SimpleClick/dataset/", "refcoco", "train")
    # for i in tqdm(range(referseg.ImgTotal)):
    #     sample = referseg.get_sample(i)
    #     if sample is None:
    #         print(f'[INFO] Empty annotation found in {i}')
    # referseg.save_missing_record()
    # print(f'[INFO] ReferSeg refcoco train dataset check done')
    # referseg = ReferSegDataset("/mnt/nas/share/home/zmz/code/SimpleClick/dataset/", "refcoco", "val")
    # for i in tqdm(range(referseg.ImgTotal)):
    #     sample = referseg.get_sample(i)
    #     if sample is None:
    #         print(f'[INFO] Empty annotation found in {i}')
    # referseg.save_missing_record()
    # print(f'[INFO] ReferSeg refcoco val dataset check done')
    # referseg = ReferSegDataset("/mnt/nas/share/home/zmz/code/SimpleClick/dataset/", "refcoco+", "train")
    # for i in tqdm(range(referseg.ImgTotal)):
    #     sample = referseg.get_sample(i)
    #     if sample is None:
    #         print(f'[INFO] Empty annotation found in {i}')
    # referseg.save_missing_record()
    # print(f'[INFO] ReferSeg refcoco+ train dataset check done')
    # referseg = ReferSegDataset("/mnt/nas/share/home/zmz/code/SimpleClick/dataset/", "refcoco+", "val")
    # for i in tqdm(range(referseg.ImgTotal)):
    #     sample = referseg.get_sample(i)
    #     if sample is None:
    #         print(f'[INFO] Empty annotation found in {i}')
    # referseg.save_missing_record()
    # print(f'[INFO] ReferSeg refcoco+ val dataset check done')
    referseg = ReferSegDataset("/mnt/nas/share/home/zmz/code/SimpleClick/dataset/", "refclef", "train")
    for i in tqdm(range(referseg.ImgTotal)):
        sample = referseg.get_sample(i)
        if sample is None:
            print(f'[INFO] Empty annotation found in {i}')
    referseg.save_missing_record()
    print(f'[INFO] ReferSeg refclef train dataset check done')
    referseg = ReferSegDataset("/mnt/nas/share/home/zmz/code/SimpleClick/dataset/", "refclef", "val")
    for i in tqdm(range(referseg.ImgTotal)):
        sample = referseg.get_sample(i)
        if sample is None:
            print(f'[INFO] Empty annotation found in {i}')
    referseg.save_missing_record()
    print(f'[INFO] ReferSeg refclef val dataset check done')
    referseg = ReferSegDataset("/mnt/nas/share/home/zmz/code/SimpleClick/dataset/", "refcocog", "train")
    for i in tqdm(range(referseg.ImgTotal)):
        sample = referseg.get_sample(i)
        if sample is None:
            print(f'[INFO] Empty annotation found in {i}')
    referseg.save_missing_record()
    print(f'[INFO] ReferSeg refcocog train dataset check done')
    referseg = ReferSegDataset("/mnt/nas/share/home/zmz/code/SimpleClick/dataset/", "refcocog", "val")
    for i in tqdm(range(referseg.ImgTotal)):
        sample = referseg.get_sample(i)
        if sample is None:
            print(f'[INFO] Empty annotation found in {i}')
    referseg.save_missing_record()
    print(f'[INFO] ReferSeg refcocog val dataset check done')
    print(f'[INFO] All dataset check done')