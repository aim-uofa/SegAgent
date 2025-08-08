import  cv2
import numpy as np
import os
import torchshow
raw_data_path = 'process_dataset/to_label'
img_list = os.listdir(raw_data_path)
from PIL import Image
import re
def create_checkerboard(height, width, square_size):
    # Create a checkerboard pattern
    checkerboard = np.indices((height // square_size, width // square_size)).sum(axis=0) % 2
    # Repeat the pattern to match the size of the image
    checkerboard = np.repeat(np.repeat(checkerboard, square_size, axis=0), square_size, axis=1)
    # Scale the pattern to the range [0, 255]
    checkerboard = (checkerboard * 128).astype(np.uint8) + 127
    # Convert the checkerboard to a color image
    checkerboard = np.stack([checkerboard, checkerboard, checkerboard], axis=-1)
    return checkerboard
for img_name in img_list:
    img_path = os.path.join(raw_data_path, img_name)
    if 'alpha' in img_path:
        continue
    if 'combined' in img_path:
        continue
    if 'rgb' in img_path:
        continue
    if 'img' in img_path:
        continue
    if 'click' in img_path:
        continue
    if 'mask' not in img_path:
        continue
    img = cv2.imread(img_path)
    print('img_shape', img.shape)
    mask_img = img
    # process_dataset/to_label/15835639065_72a6d7d0a1_oimg_mask10.jpg -> process_dataset/to_label/15835639065_72a6d7d0a1_o.jpg
    ori_img_path = re.sub(r'_mask\d+', '', img_path)

    print('resize', img_path)
    rgb_img = cv2.imread(ori_img_path)
       # 确保mask是二值图像
    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    
    # 创建棋盘背景
    checkerboard = create_checkerboard(rgb_img.shape[0], rgb_img.shape[1], 10)
    
    image = Image.open(ori_img_path)
    alpha_mask = Image.open(img_path)
    foreground = Image.new("RGBA", image.size)
    # check dimensions
    alpha_mask = alpha_mask.convert('L')
    print('image.size', image.size)
    print('alpha_mask.size', alpha_mask.size)
    foreground.paste(image, (0, 0), alpha_mask)
    combined = Image.composite(foreground, Image.fromarray(checkerboard.astype(np.uint8)), alpha_mask)
    out_path = img_path.replace('.jpg', '_combined.jpg')
    combined.save(out_path) 
    

