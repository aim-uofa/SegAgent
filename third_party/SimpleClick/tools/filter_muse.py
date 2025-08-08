import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2,os
from tqdm import tqdm
# Function to overlay text on an image
def overlay_text_on_image(image, text, position, font_size=20, font_path=None):
    # Convert to PIL image
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    
    # Load a font
    if font_path is None:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, font_size)
    
    # Add text to image
    draw.text(position, text, font=font, fill=(255, 0, 0, 255))
    
    return np.array(image_pil)

# Function to overlay mask on an image
def overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.5):
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create a color mask
    color_mask = np.zeros_like(image)
    color_mask[mask == 1] = color
    
    # Overlay the mask on the image
    overlay_image = cv2.addWeighted(image, 1, color_mask, alpha, 0)
    
    return overlay_image

# Load your dataset (update with your paths)
def load_refcoco_dataset(json_path='data/refcoco_train.json'):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data

# Convert segmentation to mask
def segmentation_to_mask(segmentation, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for segment in segmentation:
        polygon = np.array(segment).reshape(-1, 2)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return mask

# Example function to visualize a single annotation
def visualize_annotation(data_item, image_dir, output_dir):
    if 'file_name' not in data_item:
        file_name = data_item['coco_url'].split('/')[-1]
        data_item['file_name'] = file_name
        image_dir = "/home/zmz/code/SimpleClick/dataset/coco/train2017"
    image_path = f"{image_dir}/{data_item['file_name']}"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    # Load the image
    image = np.array(Image.open(image_path))
    answers = data_item['answers']
    #print(answers)
    idx = 0
    for ans in answers:
        for ann in ans:
            mask = segmentation_to_mask(ann['segmentation'], data_item['height'], data_item['width'])
            image_with_mask = overlay_mask_on_image(image, mask)
            if 'rephrased_name' in ann:
                image_with_text = overlay_text_on_image(image_with_mask, ann['rephrased_name'], position=(10, 10))
            else:
                print("No rephrased_name", file_name)
            continue
            plt.imshow(image_with_text)
            plt.axis('off')
            plt.show()
            plt.savefig(f'{output_dir}/{data_item["file_name"].split("/")[-1].split(".")[0]}_{idx}.png')
            idx += 1


# Main function
def main():
    # Load dataset
    json_path = 'data/MUSE/MUSE_train.json'
    filter_path = 'data/MUSE/MUSE_train_filter.json'
    data = load_refcoco_dataset(json_path)
    data_filter = []
    # Define your image directory
    image_dir = '/home/zmz/code/SimpleClick/dataset/refer_seg/images/mscoco/images/train2014'
    output_dir = 'output'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Visualize annotations
    with_rephrased_name = 0
    for data_item in tqdm(data):
        try:
            if 'rephrased_name' in data_item['answers'][0][0]:
                print(data_item['answers'][0][0]['rephrased_name'])
                with_rephrased_name += 1
                data_filter.append(data_item)
            else:
                print("No rephrased_name")
            #visualize_annotation(data_item, image_dir, output_dir)
        except:
            print("Error")
    print(with_rephrased_name, len(data_filter))
    with open(filter_path, 'w') as f:
        json.dump(data_filter, f)

if __name__ == '__main__':
    main()
