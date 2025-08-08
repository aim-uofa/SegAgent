import numpy as np
import torch
from PIL import Image, ImageOps

from evaltools.visual_utils import overlay_mask

try:
    from osprey.train.train import DataArguments, preprocess_multimodal

    data_args = DataArguments()
    data_args.mm_use_im_start_end = False
    data_args.is_multimodal = True

except:
    import sys

    sys.path.append("/home/zmz/code/Osprey-main")

color_map = {"green": (0, 255, 0), "red": (255, 0, 0), "blue": (0, 0, 255)}

DETAILED_QUESTIONS = [
    """
You are a highly skilled segmentation annotator. You have been provided with an image and an initial mask marked by <region> that roughly covers the object described below. Your task is to refine this mask to make it as accurate as possible. Based on the given image and the mask, perform the following actions:

1. Positive Point (x, y): Add a positive point if a part of the object is not covered by the mask. This will help expand the mask to include the missing area.

2. Negative Point (x, y): Add a negative point if an area outside the object is incorrectly included in the mask. This will help refine the mask by excluding unnecessary regions.

3. No Object Found: If you cannot locate the described object in the image, select this option. This will help us identify images that do not contain the object.

Please proceed with the action that best enhances the accuracy of the mask.

The description of the object is as follows: <description>.
"""
]
DETAILED_QUESTIONS_NO_MASK = [
    """
You are a highly skilled segmentation annotator. We have provided you with an image and an initial mask marked by a **semi-transparent green mask** that roughly covers the object described below. Your task is to refine this mask to make it as accurate as possible. Based on the given image and the mask, perform the following actions:

1. Positive Point (x, y): 
   Add a positive point if any part of the object is not covered by the mask. This will expand the mask to include the missing area.  
   *Example:* Add a positive point on any corner or edge of the object that the mask does not cover.

2. Negative Point (x, y):  
   Add a negative point if an area outside the object is incorrectly included in the mask. This will refine the mask by excluding unnecessary regions.  
   *Example:* Add a negative point where the mask extends into the background or any non-object area.

Please choose the action that best enhances the accuracy of the mask.

The description of the object is as follows: <description>.
    """
]

DETAILED_QUESTIONS_NO_MASK_FINISH = [
    """
You are a highly skilled segmentation annotator. We have provided you with an image and an initial mask marked by a **semi-transparent green mask** that roughly covers the object described below. Your task is to refine this mask to make it as accurate as possible. Based on the given image and the mask, perform the following actions:

1. Positive Point (x, y): 
   Add a positive point if any part of the object is not covered by the mask. This will expand the mask to include the missing area.  
   *Example:* Add a positive point on any corner or edge of the object that the mask does not cover.

2. Negative Point (x, y):  
   Add a negative point if an area outside the object is incorrectly included in the mask. This will refine the mask by excluding unnecessary regions.  
   *Example:* Add a negative point where the mask extends into the background or any non-object area.

3. Finish:
    If you have refined the mask to the best of your ability, select this option. This will help us identify images that have been accurately annotated.

Please choose the action that best enhances the accuracy of the mask.

The description of the object is as follows: <description>.
    """
]

REFINE_PROMPT = [
    "Please refine the semi-transparent green mask of <description>.",
    "Please refine this semi-transparent green mask.",
]

# LOCATE_PROMPT = [
#     "Please locate <description> in the image.",
#     "Where is <description> located in the image?",
# ]


def get_init_inputs(
    img_path,
    processor,
    pred_bboxes,
    mask,
    round_ids=0,
    last_round_source=None,
    mask_overlay=False,
    mask_color="green",
):
    if round_ids == 0:
        image = Image.open(img_path).convert("RGB")
        if image.size[0] != mask.shape[1] or image.size[1] != mask.shape[0]:
            # raise ValueError(f'Image size {image.size} does not match mask size {mask.shape}, img_path={img_path}')
            image = ImageOps.exif_transpose(image).convert("RGB")
        if mask_overlay:
            image = overlay_mask(np.array(image), mask[0], color=color_map[mask_color])
            image = Image.fromarray(image)

        image = processor.preprocess(image, do_center_crop=False, return_tensors="pt")[
            "pixel_values"
        ][0]

        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=False
        ).squeeze(0)

    else:
        raise ValueError("round_ids should be 0")
        # image = last_round_source["image"]

    cur_token_len = (image.shape[1] // 16) * (image.shape[2] // 16)

    mask = mask.to(image.device)

    begin_str = """<image>.\nThis provides an overview of the picture.\n"""

    sources = dict()
    sources["conversations"] = []
    question = "Please give me a short description of <mask><pos>."

    sources["conversations"].append({"from": "human", "value": begin_str + question})

    sources = preprocess_multimodal(
        [sources["conversations"]], data_args, cur_token_len
    )

    data_dict = {}
    data_dict["sources"] = sources
    data_dict["image"] = image
    data_dict["masks"] = mask

    return data_dict
