import  cv2
import numpy as np
import os
import torchshow
raw_data_path = 'data/epoch_120'
img_list = os.listdir(raw_data_path)
from PIL import Image
def create_checkerboard(height, width, square_size):
    # Create a checkerboard pattern
    checkerboard = np.indices((height // square_size, width // square_size)).sum(axis=0) % 2
    # Repeat the pattern to match the size of the image
    checkerboard = np.repeat(np.repeat(checkerboard, square_size, axis=0), square_size, axis=1)
    # Scale the pattern to the range [0, 255]
    checkerboard = (checkerboard * 255).astype(np.uint8)
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
    img = cv2.imread(img_path)
    print('img_shape', img.shape)
    #split the image 左边的是原图，右边的是alpha mask
    img = np.split(img, 2, axis=1)
    rgb_img = img[0]
    alpha_img = img[1]
    # cv2.imwrite(img_path.replace('.jpg', '_rgb.jpg'), rgb_img)
    # cv2.imwrite(img_path.replace('.jpg', '_alpha.jpg'), alpha_img)
    print('resize', img_path)
    # 进行抠图 
    # Create a checkerboard background
    checkerboard = create_checkerboard(rgb_img.shape[0], rgb_img.shape[1], 10)
    # Use the alpha mask to extract the foreground from the image
    image_path = img_path.replace('.jpg', '_rgb.jpg')
    alpha_mask_path = img_path.replace('.jpg', '_alpha.jpg')
    image = Image.open(image_path)
    alpha_mask = Image.open(alpha_mask_path)
    foreground = Image.new("RGBA", image.size)
    # check dimensions
    alpha_mask = alpha_mask.convert('L')
    print('image.size', image.size)
    print('alpha_mask.size', alpha_mask.size)
    foreground.paste(image, (0, 0), alpha_mask)
    combined = Image.composite(foreground, Image.fromarray(checkerboard.astype(np.uint8)), alpha_mask)
    out_path = img_path.replace('.jpg', '_combined.jpg')
    combined.save(out_path) 

