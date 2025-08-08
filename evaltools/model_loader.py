import torch
from pathlib import Path
import sys
sys.path.append('../')

# --- 依赖项导入 ---
import os
import cv2
from third_party.SimpleClick.isegm.inference.clicker import (
    Click,
    Clicker,
    Clicker_sampler,
)
import time
import numpy as np
from PIL import Image
from evaltools.visual_utils import (
    visualize_mask_and_point,
    overlay_points,
    overlay_boxes,
    visualize_mask_and_pointlist,
    overlay_mask,
)
from PIL import Image,ImageOps
# from isegm.inference import utils
# from isegm.utils.exp import load_config_file
# from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference
# from isegm.inference.predictors import get_predictor
# from isegm.inference.evaluation import evaluate_dataset
from third_party.SimpleClick.isegm.inference import utils
from third_party.SimpleClick.isegm.utils.exp import load_config_file
from third_party.SimpleClick.isegm.model.modeling.pos_embed import interpolate_pos_embed_inference
from third_party.SimpleClick.isegm.inference.predictors import get_predictor
from third_party.SimpleClick.isegm.inference.evaluation import evaluate_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM



color_map = {
    'green': (0, 255, 0),
    'red': (255, 0, 0),
    'blue': (0, 0, 255)
}

def load_segmentation_model(args):
    """
    Loads the appropriate segmentation model based on arguments.
    """
    if args.seg_model == 'simple_click':
        cfg = load_config_file(args.config_path, return_edict=True)
        cfg.EXPS_PATH = Path(cfg.EXPS_PATH)
        checkpoints_list, _, _ = get_checkpoints_list_and_logs_path(args, cfg)
        checkpoint_path = checkpoints_list[0]
        model = utils.load_is_model(checkpoint_path, args.device, args.eval_ritm)

        dataset_name = 'none'
        predictor_params, zoomin_params = get_predictor_and_zoomin_params(
            args, dataset_name, eval_ritm=args.eval_ritm
        )

        # For SimpleClick models, interpolate the positional embedding
        if not args.eval_ritm and zoomin_params:
            interpolate_pos_embed_inference(
                model.backbone, zoomin_params['target_size'], args.device
            )

        predictor = get_predictor(
            model,
            args.mode,
            args.device,
            prob_thresh=args.thresh,
            predictor_params=predictor_params,
            zoom_in_params=zoomin_params
        )
        predictor.visualize = args.visualize
        predictor.record_trace = args.record_trace
        predictor.start_index = args.start_index
        predictor.end_index = args.end_index if args.end_index != -1 else 999999
        segmentation_model = SegmentationModel(predictor)

    elif args.seg_model == 'sam':
        segmentation_model = SAMModel('b', args)
    elif args.seg_model == 'sam_l':
        segmentation_model = SAMModel('l', args)
    elif args.seg_model == 'sam_h':
        segmentation_model = SAMModel('h', args)
    else:
        raise ValueError(f"Unknown segmentation model: {args.seg_model}")

    return segmentation_model


def load_grounding_model(args):
    """
    Loads the appropriate grounding model based on arguments.
    """

    if 'qwen-full' in args.grounding_model:
        grounding_model = GroundingModel_Qwen_full(args.model, args)
    elif 'qwen-lora' in args.grounding_model:
        grounding_model = GroundingModel_Qwen_lora(args.model, args)
    else:
        raise ValueError(f"Unknown grounding model: {args.grounding_model}")

    return grounding_model

def load_model(args):
    segmentation_model = load_segmentation_model(args)
    grounding_model = load_grounding_model(args)
    return segmentation_model, grounding_model

def get_checkpoints_list_and_logs_path(args, cfg):
    logs_prefix = ''
    if args.exp_path:
        rel_exp_path = args.exp_path
        checkpoint_prefix = ''
        if ':' in rel_exp_path:
            rel_exp_path, checkpoint_prefix = rel_exp_path.split(':')

        exp_path_prefix = cfg.EXPS_PATH / rel_exp_path
        candidates = list(exp_path_prefix.parent.glob(exp_path_prefix.stem + '*'))
        assert len(candidates) == 1, "Invalid experiment path."
        exp_path = candidates[0]
        checkpoints_list = sorted((exp_path / 'checkpoints').glob(checkpoint_prefix + '*.pth'), reverse=True)
        assert len(checkpoints_list) > 0, "Couldn't find any checkpoints."

        if checkpoint_prefix:
            if len(checkpoints_list) == 1:
                logs_prefix = checkpoints_list[0].stem
            else:
                logs_prefix = f'all_{checkpoint_prefix}'
        else:
            logs_prefix = 'all_checkpoints'

        logs_path = args.logs_path / exp_path.relative_to(cfg.EXPS_PATH)
    else:
        checkpoints_list = [Path(utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint))]
        # logs_path = args.logs_path / 'others' / checkpoints_list[0].stem
        logs_path = f'./logs/{checkpoints_list[0].stem}'

    return checkpoints_list, logs_path, logs_prefix


def get_predictor_and_zoomin_params(args, dataset_name, apply_zoom_in=True, eval_ritm=False):
    predictor_params = {}

    if args.clicks_limit is not None:
        if args.clicks_limit == -1:
            args.clicks_limit = args.n_clicks
        predictor_params['net_clicks_limit'] = args.clicks_limit

    zoom_in_params = None
    if apply_zoom_in and eval_ritm:
        if args.eval_mode == 'cvpr':
            zoom_in_params = {
                'target_size': 600 if dataset_name == 'DAVIS' else 400
            }
        elif args.eval_mode.startswith('fixed'):
            crop_size = int(args.eval_mode[5:])
            zoom_in_params = {
                'skip_clicks': -1,
                'target_size': (crop_size, crop_size)
            }
        else:
            raise NotImplementedError

    if apply_zoom_in and not eval_ritm:
        if args.eval_mode == 'cvpr':
            zoom_in_params = {
                'skip_clicks': -1,
                'target_size': (672, 672) if dataset_name == 'DAVIS' else (448, 448)
            }
        elif args.eval_mode.startswith('fixed'):
            crop_size = args.eval_mode.split(',')
            crop_size_h = int(crop_size[0][5:])
            crop_size_w = crop_size_h
            if len(crop_size) == 2:
                crop_size_w = int(crop_size[1])
            zoom_in_params = {
                'skip_clicks': -1,
                'target_size': (crop_size_h, crop_size_w)
            }
        else:
            raise NotImplementedError

    return predictor_params, zoom_in_params


class SegmentationModel:
    def __init__(self, predictor):
        self.predictor = predictor

    def set_input_image(self, image):
        if self.predictor is not None:
            self.predictor.set_input_image(image)

    def get_prediction(self, clicker, box=None, mask=None):
        pred_mask = self.predictor.get_prediction(clicker)
        return pred_mask > 0.49

    def image_process(self,img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def release_resources(self):
        del self.predictor
        torch.cuda.empty_cache()
        self.predictor = None

    def predict_clicks_from_mask(self,target_mask, file_name=None, object_id=None, click_num=2, pred_thr=0.49):
        """
        This function is only used for SimpleClick,
        which is used for predicting initial clicks from mask generated by gtbox and SAM.
        This is an inverse process.
        """
        predictor = self.predictor
        pred_mask = np.zeros_like(target_mask)
        clicker = Clicker(gt_mask=target_mask)
        for _ in range(click_num):
            clicker.object_id = object_id
            clicker.make_next_click(pred_mask, file_name) 
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr 

        return clicker.get_clicks(), pred_mask


from segment_anything import SamPredictor, sam_model_registry

def get_points_nd(clicks_list):

    points, labels = [], []
    for click in clicks_list:
        h, w = click.coords_and_indx[:2]
        points.append([w, h])
        labels.append(int(click.is_positive))
    return np.array(points), np.array(labels)
    
class SAMModel:
    def __init__(self,model_size='b',args=None):
        # weights/sam/sam_vit_b_01ec64.pth
        # weights/sam/sam_vit_h_4b8939.pth
        # weights/sam/sam_vit_l_0b3195.pth
        self.args = args
        if self.args.grounding_model == 'qwen-lora-box' or self.args.grounding_model == 'qwen-full-box':
            self.predictor = None
        else:
            if model_size == 'b':
                sam_path = '/home/zmz/code/SegNext/weights/sam/sam_vit_b_01ec64.pth'
                if os.path.exists(sam_path):
                    pass
                else:
                    sam_path = args.checkpoint
                sam = sam_model_registry["vit_b"](checkpoint=sam_path)
                device = torch.device('cuda:0')
                sam.to(device)
            elif model_size == 'h':
                sam_path = '/home/zmz/code/SegNext/weights/sam/sam_vit_h_4b8939.pth'
                if os.path.exists(sam_path):
                    pass
                else:
                    sam_path = args.checkpoint
                sam = sam_model_registry["vit_h"](checkpoint=sam_path)
                device = torch.device('cuda:0')
                sam.to(device)
            elif model_size == 'l':
                sam_path = '/home/zmz/code/SegNext/weights/sam/sam_vit_l_0b3195.pth'
                if os.path.exists(sam_path):
                    pass
                else:
                    sam_path = args.checkpoint
                sam = sam_model_registry["vit_l"](checkpoint=sam_path)
                device = torch.device('cuda:0')
                sam.to(device)
            self.predictor = SamPredictor(sam)
        self.pred_thres = 0.50

    def set_input_image(self, image):
        if self.predictor is not None:
            self.predictor.set_image(image)

    def get_prediction(self, clicker=None, box=None, mask=None):
        pred_logits = mask
        if clicker is not None:
            if box is not None:
                box = np.array(box)
            clicks_list = clicker.get_clicks()
            points_nd, labels_nd = get_points_nd(clicks_list)
            preds, scores, pred_logits = self.predictor.predict(
                    points_nd, labels_nd, mask_input=pred_logits, box=box)
            max_score_idx = np.argmax(scores)
            pred_mask = preds[max_score_idx] > self.pred_thres
            pred_logits = pred_logits[[max_score_idx]]
            return (pred_mask, pred_logits)
        if box is not None:
            # box (np.ndarray or None): A length 4 array given a box prompt to the
            # model, in XYXY format.
            box = np.array(box)
            preds, scores, pred_logits = self.predictor.predict(
                box=box, mask_input=pred_logits)
            max_score_idx = np.argmax(scores)
            pred_mask = preds[max_score_idx] > self.pred_thres
            pred_logits = pred_logits[[max_score_idx]]


        return (pred_mask, pred_logits)

    def image_process(self,img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def release_resources(self):
        del self.predictor
        torch.cuda.empty_cache()
        self.predictor = None

import re


class GroundingModel_Qwen_full(): 
    def __init__(self, model_path, args):
        self.workspace = os.environ.get('VIS_DIR', os.getcwd())
        self.use_mask_module = args.use_mask_module
        self.visualize = args.visualize
        self.args = args
        #model_path = '/home/zmz/code/Osprey-main/weights/Qwen-VL-chat'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map='cuda', trust_remote_code=True).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token_id = self.tokenizer.eod_id

        self.prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPicture 1: <img>{}</img>\nPlease optimize the semi-transparent green mask in a point-wise manner based on the image and description content, so that it covers the target object as accurately as possible. The object description is as follows:<ref>{}</ref><|im_end|>\n<|im_start|>assistant\n'
        self.box_predict = True if 'box' in args.grounding_model else False
        self.predictor = None

    def build_prompt(self, init_inputs, last_ref_box_str=None):

        prompt = self.prompt.format(init_inputs['img_path'],
                                    init_inputs['caption'][0])

        if last_ref_box_str:
            prompt += last_ref_box_str
        return prompt, None

    def generate_response(self, prompt, image=None, masks=None, conv=None):
        # save a new tmp image with mask overlay
        # 
        img_path = image 
        mask_overlay = True
        mask_color = 'green'
        image = Image.open(img_path).convert('RGB')
        if image.size[0] != masks.shape[1] or image.size[1] != masks.shape[0]:
            #raise ValueError(f'Image size {image.size} does not match mask size {mask.shape}, img_path={img_path}')
            image = ImageOps.exif_transpose(image).convert('RGB')
        if mask_overlay:
            image = overlay_mask(np.array(image), masks[0],color=color_map[mask_color])
            image = Image.fromarray(image)
        #tmp_path = os.path.join(self.workspace, img_path.split('/')[-1])
        random_num = time.time()
        tmp_path = os.path.join(self.workspace, img_path.split('/')[-1].replace('.jpg', f'_{random_num}.jpg'))
        image.save(tmp_path)
        prompt = prompt.replace(img_path, tmp_path)
        token_result = self.tokenizer([prompt],
                                      return_tensors='pt',
                                      padding='longest')
        input_ids = token_result.input_ids  # print(self.tokenizer.decode(input_ids[0]))
        
        # attention_mask = token_result.attention_mask
        # pred = self.model.generate(
        #     input_ids=input_ids.cuda(),
        #     attention_mask=attention_mask.cuda(),
        #     do_sample=False,
        #     num_beams=1,
        #     max_new_tokens=28,
        #     min_new_tokens=10,
        #     length_penalty=1,
        #     num_return_sequences=1,
        #     use_cache=True,
        #     pad_token_id=self.tokenizer.eod_id,
        #     eos_token_id=self.tokenizer.eod_id,
        #     #masks_ids = mask_token
        # )
        generation_config = self.model.generation_config
        stop_words_ids = [[self.tokenizer.im_end_id], [self.tokenizer.im_start_id]]
        pred = self.model.generate(
                    input_ids.cuda(),
                    stop_words_ids=stop_words_ids,
                    return_dict_in_generate=False,
                    generation_config=generation_config,
                    do_sample=False,
                )
        answers = [
            self.tokenizer.decode(_[input_ids.size(1):].cpu(),
                                  skip_special_tokens=False) for _ in pred
        ]
        # remove the tmp image
        os.remove(tmp_path)

        return answers[0]

    def process_response(self, outputs):
        # ' Positive point: (200, 534)'
        try:
            if 'Positive point' in outputs:
                is_positive = True
            else :
                assert 'Negative point' in outputs
                is_positive = False
            outputs_point = outputs.split('point: ')[-1]
            point = re.findall(r'\((.*?)\)', outputs_point)
            point = [int(x) for x in point[0].split(',')]
            relative_coor = (point[0] / 1000, point[1] / 1000)
        except:
            is_positive = None
            relative_coor = None
        # <ref>person on far left in white jacket</ref><box>(249,123),(534,496)</box> Positive point: (381, 337)
        if self.box_predict:
            try:
                try:
                    if 'Positive point' in outputs:
                        is_positive = True
                    else :
                        assert 'Negative point' in outputs
                        is_positive = False
                    outputs_point = outputs.split('point: ')[-1]
                    point = re.findall(r'\((.*?)\)', outputs_point)
                    point = [int(x) for x in point[0].split(',')]
                    relative_coor = (point[0] / 1000, point[1] / 1000)
                except:
                    is_positive = None
                    relative_coor = None
                try:
                    outputs_box = outputs.split('<box>')[-1].split('</box>')[0]
                    PATTERN = re.compile(r'\((.*?)\),\((.*?)\)')
                    predict_bbox = re.findall(PATTERN, outputs_box)
                    try:
                        if ',' not in predict_bbox[0][0] or ',' not in predict_bbox[0][
                                1]:
                            predict_bbox = (0., 0., 0., 0.)
                        else:
                            x1, y1 = [
                                float(tmp) for tmp in predict_bbox[0][0].split(',')
                            ]
                            x2, y2 = [
                                float(tmp) for tmp in predict_bbox[0][1].split(',')
                            ]
                            predict_bbox = (x1, y1, x2, y2)
                    except:
                        predict_bbox = (0., 0., 0., 0.)
                    box = predict_bbox
                except:
                    box = None
            except:
                is_positive = None
                relative_coor = None
                box = None
            return is_positive, relative_coor, box

        return is_positive, relative_coor


    def release_resources(self):
        # 删除模型和分词器的引用
        del self.model
        del self.tokenizer

        # 释放未使用的CUDA内存
        torch.cuda.empty_cache()



class GroundingModel_Qwen_lora():

    def __init__(self, model_path, args):
        self.workspace = os.environ.get('VIS_DIR', os.getcwd())
        self.use_mask_module = args.use_mask_module
        self.visualize = args.visualize
        self.args = args
        #model_path = '/home/zmz/code/Osprey-main/weights/Qwen-VL-chat'
        from peft import AutoPeftModelForCausalLM
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, device_map='cuda', trust_remote_code=True).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        self.prompt0 = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPicture 1: <img>{}</img>\nPlease optimize the semi-transparent green mask in a point-wise manner based on the image and description content, so that it covers the target object as accurately as possible. The object description is as follows:<ref>{}</ref><|im_end|>\n<|im_start|>assistant\n'
        self.prompt = 'Please optimize the semi-transparent green mask in a point-wise manner based on the image and description content, so that it covers the target object as accurately as possible. The object description is as follows:<ref>{}</ref>'
        self.predictor = None
        self.box_predict = True if 'box' in args.grounding_model else False

    def build_prompt2(self, init_inputs, last_ref_box_str=None):
        # prompt = self.prompt.format(init_inputs['img_path'],
        #                             init_inputs['caption'][0])

        prompt = self.prompt.format(init_inputs['caption'][0])

        if last_ref_box_str:
            prompt = last_ref_box_str + prompt

        return prompt, None

    def build_prompt(self, init_inputs, last_ref_box_str=None):

        prompt = self.prompt0.format(init_inputs['img_path'],
                                    init_inputs['caption'][0])

        if last_ref_box_str:
            prompt += last_ref_box_str
        return prompt, None

    def generate_response2(self, prompt, image=None, masks=None, conv=None):
        # save a new tmp image with mask overlay
        # 
        img_path = image 
        mask_overlay = True
        mask_color = 'green'
        image = Image.open(img_path).convert('RGB')
        if image.size[0] != masks.shape[1] or image.size[1] != masks.shape[0]:
            #raise ValueError(f'Image size {image.size} does not match mask size {mask.shape}, img_path={img_path}')
            image = ImageOps.exif_transpose(image).convert('RGB')
        if mask_overlay:
            image = overlay_mask(np.array(image), masks[0],color=color_map[mask_color])
            image = Image.fromarray(image)
        #tmp_path = os.path.join(self.workspace, img_path.split('/')[-1])
        random_num = time.time()
        tmp_path = os.path.join(self.workspace, img_path.split('/')[-1].replace('.jpg', f'_{random_num}.jpg'))
        image.save(tmp_path)
        query = self.tokenizer.from_list_format([
            {
                'image': tmp_path,  # Either a local path or an url
            },  # Either a local path or an url
            {
                'text': prompt,
            },
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None) 
        # prompt = prompt.replace(img_path, tmp_path)
        # token_result = self.tokenizer([prompt],
        #                               return_tensors='pt',
        #                               padding='longest')
        # input_ids = token_result.input_ids  # print(self.tokenizer.decode(input_ids[0]))
        
        # attention_mask = token_result.attention_mask
        # pred = self.model.generate(
        #     input_ids=input_ids.cuda(),
        #     attention_mask=attention_mask.cuda(),
        #     do_sample=False,
        #     num_beams=1,
        #     max_new_tokens=28,
        #     min_new_tokens=10,
        #     length_penalty=1,
        #     num_return_sequences=1,
        #     use_cache=True,
        #     pad_token_id=self.tokenizer.eod_id,
        #     eos_token_id=self.tokenizer.eod_id,
        #     #masks_ids = mask_token
        # )
        # answers = [
        #     self.tokenizer.decode(_[input_ids.size(1):].cpu(),
        #                           skip_special_tokens=False) for _ in pred
        # ]
        os.remove(tmp_path)

        return response

    def generate_response(self, prompt, image=None, masks=None, conv=None):
        # save a new tmp image with mask overlay
        # 
        img_path = image 
        mask_overlay = True
        mask_color = 'green'
        image = Image.open(img_path).convert('RGB')
        if image.size[0] != masks.shape[1] or image.size[1] != masks.shape[0]:
            #raise ValueError(f'Image size {image.size} does not match mask size {mask.shape}, img_path={img_path}')
            image = ImageOps.exif_transpose(image).convert('RGB')
        if mask_overlay:
            image = overlay_mask(np.array(image), masks[0],color=color_map[mask_color])
            image = Image.fromarray(image)
        #tmp_path = os.path.join(self.workspace, img_path.split('/')[-1])
        random_num = time.time()
        tmp_path = os.path.join(self.workspace, img_path.split('/')[-1].replace('.jpg', f'_{random_num}.jpg'))
        image.save(tmp_path)
        prompt = prompt.replace(img_path, tmp_path)
        token_result = self.tokenizer([prompt],
                                      return_tensors='pt',
                                      padding='longest')
        input_ids = token_result.input_ids  # print(self.tokenizer.decode(input_ids[0]))
        
        # attention_mask = token_result.attention_mask
        # pred = self.model.generate(
        #     input_ids=input_ids.cuda(),
        #     attention_mask=attention_mask.cuda(),
        #     do_sample=False,
        #     num_beams=1,
        #     max_new_tokens=28,
        #     min_new_tokens=10,
        #     length_penalty=1,
        #     num_return_sequences=1,
        #     use_cache=True,
        #     pad_token_id=self.tokenizer.eod_id,
        #     eos_token_id=self.tokenizer.eod_id,
        #     #masks_ids = mask_token
        # )
        generation_config = self.model.generation_config
        stop_words_ids = [[self.tokenizer.im_end_id], [self.tokenizer.im_start_id]]
        pred = self.model.model.generate(
                    input_ids.cuda(),
                    stop_words_ids=stop_words_ids,
                    return_dict_in_generate=False,
                    generation_config=generation_config,
                    do_sample=False,
                )
        answers = [
            self.tokenizer.decode(_[input_ids.size(1):].cpu(),
                                  skip_special_tokens=False) for _ in pred
        ]
        # remove the tmp image
        os.remove(tmp_path)

        return answers[0]

    def process_response(self, outputs):
        # ' Positive point: (200, 534)'
        try:
            if 'Positive point' in outputs:
                is_positive = True
            else :
                is_positive = False
            point = re.findall(r'\((.*?)\)', outputs)
            point = [int(x) for x in point[0].split(',')]
            relative_coor = (point[0] / 1000, point[1] / 1000)
        except:
            is_positive = None
            relative_coor = None

        # <ref>person on far left in white jacket</ref><box>(249,123),(534,496)</box> Positive point: (381, 337)
        if self.box_predict:
            try:
                try:
                    if 'Positive point' in outputs:
                        is_positive = True
                    else :
                        assert 'Negative point' in outputs
                        is_positive = False
                    outputs_point = outputs.split('point: ')[-1]
                    point = re.findall(r'\((.*?)\)', outputs_point)
                    point = [int(x) for x in point[0].split(',')]
                    relative_coor = (point[0] / 1000, point[1] / 1000)
                except:
                    is_positive = None
                    relative_coor = None
                try:
                    outputs_box = outputs.split('<box>')[-1].split('</box>')[0]
                    PATTERN = re.compile(r'\((.*?)\),\((.*?)\)')
                    predict_bbox = re.findall(PATTERN, outputs_box)
                    try:
                        if ',' not in predict_bbox[0][0] or ',' not in predict_bbox[0][
                                1]:
                            predict_bbox = (0., 0., 0., 0.)
                        else:
                            x1, y1 = [
                                float(tmp) for tmp in predict_bbox[0][0].split(',')
                            ]
                            x2, y2 = [
                                float(tmp) for tmp in predict_bbox[0][1].split(',')
                            ]
                            predict_bbox = (x1, y1, x2, y2)
                    except:
                        predict_bbox = (0., 0., 0., 0.)
                    box = predict_bbox
                except:
                    box = None
            except:
                is_positive = None
                relative_coor = None
                box = None
            return is_positive, relative_coor, box

        return is_positive, relative_coor


    def release_resources(self):
        # 删除模型和分词器的引用
        del self.model
        del self.tokenizer

        # 释放未使用的CUDA内存
        torch.cuda.empty_cache()