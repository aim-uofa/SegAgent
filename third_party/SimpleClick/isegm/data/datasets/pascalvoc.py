import pickle as pkl
from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_class_names_from_color(mask_path, color_to_class_name):
    # 读取图像
    image = cv2.imread(mask_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 获取唯一的颜色
    unique_colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)

    # 查找颜色对应的类别
    class_names = [color_to_class_name[tuple(color)] for color in unique_colors if tuple(color) in color_to_class_name]

    return class_names

# 获取 PASCAL VOC 颜色映射表
cmap = color_map()

# 定义颜色到类别名称的映射表
color_to_class_name = {
    tuple(cmap[1]): "aeroplane",
    tuple(cmap[2]): "bicycle",
    tuple(cmap[3]): "bird",
    tuple(cmap[4]): "boat",
    tuple(cmap[5]): "bottle",
    tuple(cmap[6]): "bus",
    tuple(cmap[7]): "car",
    tuple(cmap[8]): "cat",
    tuple(cmap[9]): "chair",
    tuple(cmap[10]): "cow",
    tuple(cmap[11]): "diningtable",
    tuple(cmap[12]): "dog",
    tuple(cmap[13]): "horse",
    tuple(cmap[14]): "motorbike",
    tuple(cmap[15]): "person",
    tuple(cmap[16]): "pottedplant",
    tuple(cmap[17]): "sheep",
    tuple(cmap[18]): "sofa",
    tuple(cmap[19]): "train",
    tuple(cmap[20]): "tvmonitor"
}

# class PascalVocDataset(ISDataset):
#     def __init__(self, dataset_path, split='train', **kwargs):
#         super().__init__(**kwargs)
#         assert split in {'train', 'val', 'trainval', 'test'}

#         self.dataset_path = Path(dataset_path)
#         self._images_path = self.dataset_path / "JPEGImages"
#         self._insts_path = self.dataset_path / "SegmentationObject"
#         self.dataset_split = split

#         if split == 'test':
#             with open(self.dataset_path / f'ImageSets/Segmentation/test.pickle', 'rb') as f:
#                 self.dataset_samples, self.instance_ids = pkl.load(f)
#         else:
#             with open(self.dataset_path / f'ImageSets/Segmentation/{split}.txt', 'r') as f:
#                 self.dataset_samples = [name.strip() for name in f.readlines()]

#     def get_sample(self, index) -> DSample:
#         sample_id = self.dataset_samples[index]
#         image_path = str(self._images_path / f'{sample_id}.jpg')
#         mask_path = str(self._insts_path / f'{sample_id}.png')

#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         instances_mask = cv2.imread(mask_path)
#         instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)
#         if self.dataset_split == 'test':
#             instance_id = self.instance_ids[index]
#             mask = np.zeros_like(instances_mask)
#             mask[instances_mask == 220] = 220  # ignored area
#             mask[instances_mask == instance_id] = 1
#             objects_ids = [1]
#             instances_mask = mask
#         else:
#             objects_ids = np.unique(instances_mask)
#             objects_ids = [x for x in objects_ids if x != 0 and x != 220]

#         return DSample(image, instances_mask, objects_ids=objects_ids, ignore_ids=[220], sample_id=index)
    
import numpy as np
import cv2
from pathlib import Path

class PascalVocDataset(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super().__init__(**kwargs)
        assert split in {'train', 'val', 'trainval', 'test'}

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / "JPEGImages"
        self._insts_path = self.dataset_path / "SegmentationObject"
        self._classes_path = self.dataset_path / "SegmentationClass"
        self.dataset_split = split

        # 类别颜色到类别名称的映射
        self.color_to_class_name = color_to_class_name  

        if split == 'test':
            with open(self.dataset_path / f'ImageSets/Segmentation/test.pickle', 'rb') as f:
                self.dataset_samples, self.instance_ids = pkl.load(f)
        else:
            with open(self.dataset_path / f'ImageSets/Segmentation/{split}.txt', 'r') as f:
                self.dataset_samples = [name.strip() for name in f.readlines()]

    def get_sample(self, index) -> DSample:
        sample_id = self.dataset_samples[index]
        image_path = str(self._images_path / f'{sample_id}.jpg')
        mask_path = str(self._insts_path / f'{sample_id}.png')
        class_mask_path = str(self._classes_path / f'{sample_id}.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取实例掩码
        instances_mask = cv2.imread(mask_path)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)

        # 读取类别掩码
        class_mask = cv2.imread(class_mask_path)
        class_mask = cv2.cvtColor(class_mask, cv2.COLOR_BGR2RGB)

        instance_to_class = {}

        if self.dataset_split == 'test':
            instance_id = self.instance_ids[index]
            mask = np.zeros_like(instances_mask)
            mask[instances_mask == 220] = 220  # 忽略区域
            mask[instances_mask == instance_id] = 1
            objects_ids = [1]
            instances_mask = mask
            # 获取对应的类别
            instance_color = tuple(np.unique(class_mask[instances_mask == instance_id], axis=0)[0])
            class_name = self.color_to_class_name.get(instance_color, "unknown")
            instance_to_class[instance_id] = class_name
        else:
            objects_ids = np.unique(instances_mask)
            objects_ids = [x for x in objects_ids if x != 0 and x != 220]
            # 映射实例 ID 到类别名称
            for obj_id in objects_ids:
                instance_color = tuple(np.unique(class_mask[instances_mask == obj_id], axis=0)[0])
                class_name = self.color_to_class_name.get(instance_color, "unknown")
                instance_to_class[obj_id] = class_name

        return DSample(image, instances_mask, objects_ids=objects_ids, ignore_ids=[220], 
                        sample_id=index, instance_to_class=instance_to_class)

