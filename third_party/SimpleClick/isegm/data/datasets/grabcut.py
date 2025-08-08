from pathlib import Path

import cv2
import numpy as np
import sys
sys.path.append('/home/zmz/code/SimpleClick')
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class GrabCutDataset(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='data_GT', masks_dir_name='boundary_GT',
                 **kwargs):
        super(GrabCutDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split('.')[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        instances_mask[instances_mask == 128] = -1
        instances_mask[instances_mask > 128] = 1

        return DSample(image, instances_mask, objects_ids=[1], ignore_ids=[-1], sample_id=index)

import json
class ReasonSegDataset(ISDataset):
    def __init__(self, dataset_path,
                 split='train',
                    **kwargs):
        super(ReasonSegDataset, self).__init__(**kwargs)
        dataset_path = Path(dataset_path)
        self.split = split
        # dataset/reason_seg/ReasonSeg/train/3588328_892066223b_o.json
        # dataset/reason_seg/ReasonSeg/train/3588328_892066223b_o.jpg
        self._images_path = dataset_path / 'ReasonSeg' / split
        self._insts_path = dataset_path / 'ReasonSeg' / split
        self.dataset_samples = [x.stem for x in sorted(self._images_path.glob('*.jpg'))]
    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / (image_name + '.jpg'))
        json_path = str(self._insts_path / (image_name + '.json'))
        image = cv2.imread(image_path)
        with open(json_path, "r") as file:
            content = json.load(file)   
        polygon = content['shapes'][0]['points']
        instances_mask = self.get_mask_from_polygon(polygon, image)
        return DSample(image, instances_mask, objects_ids=[1], ignore_ids=[0], sample_id=index)


    @staticmethod
    def get_mask_from_polygon(poly, image):
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
        poly = np.array(poly, np.int32)
        cv2.fillPoly(mask, [poly], 1)
        return mask


# Test the ReasonSegDataset

# dataset = ReasonSegDataset('/home/zmz/code/SimpleClick/dataset/reason_seg', split='train')

# sample = dataset.get_sample(0)