import json
import os

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools import mask as maskUtils
from tqdm import tqdm

from evaltools.prompt_const import (
    DETAILED_QUESTIONS,
    DETAILED_QUESTIONS_NO_MASK,
    DETAILED_QUESTIONS_NO_MASK_FINISH,
    REFINE_PROMPT,
    # LOCATE_PROMPT,
)
from evaltools.utils import AverageMeter, Summary, intersectionAndUnionGPU
from evaltools.visual_utils import (
    visualize_mask_and_point,
    # overlay_points,
    # overlay_boxes
    visualize_mask_and_pointlist,
)
from third_party.SimpleClick.isegm.inference.clicker import (
    Click,
    Clicker,
    Clicker_sampler,
)
from evaltools.prompt_const import get_init_inputs


def convert_mask_to_coco_format(mask):
    # convert gt_mask to coco format
    print("converting mask to coco format")
    if isinstance(mask, torch.Tensor):
        print("mask is tensor")
        mask = mask.cpu().numpy()
    # check mask shape
    if mask.ndim == 3:
        mask = mask[0]
    mask = mask.astype(np.uint8)
    mask = mask_util.encode(np.array(mask, order="F"))
    if isinstance(mask, list):
        mask = mask_util.merge(mask)
    if isinstance(mask, dict):
        mask = {"counts": mask["counts"].decode("utf-8"), "size": mask["size"]}
    return mask


class REFCOCOG_EVAL:
    def __init__(self, groundingmodel, segmodel, args):
        self.begin_str = "<image>\nThis provides an overview of the picture.\n"
        self.groundingmodel = groundingmodel
        self.segmodel = segmodel
        self.args = args
        self.workspace = os.environ.get("VIS_DIR", os.getcwd())
        self.image_processor = (
            self.groundingmodel.image_processor
            if hasattr(self.groundingmodel, "image_processor")
            else None
        )
        self.tokenizer = (
            self.groundingmodel.tokenizer
            if hasattr(self.groundingmodel, "tokenizer")
            else None
        )
        self.visualize = args.visualize

    def process_mask(self, masks):
        # NxHxW -> Nx512x512
        masks = masks.float()
        masks = torch.nn.functional.interpolate(
            masks.unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=False
        ).squeeze(0)
        return masks

    def set_dataset_name(self, ann_file):
        if "refclef" in ann_file:
            self.dataset_name = "refclef"
        elif "refcoco+" in ann_file:
            self.dataset_name = "refcoco+"
        elif "refcocog" in ann_file:
            self.dataset_name = "refcocog"
        elif "refcoco" in ann_file:
            self.dataset_name = "refcoco"
        elif "VOC" in ann_file:
            self.dataset_name = "pascal_voc"
        elif "LVIS" in ann_file:
            self.dataset_name = "lvis"
        elif "Reason" in ann_file:
            self.dataset_name = "reasonseg"
        elif "FSS" in ann_file:
            self.dataset_name = "fss"
        elif "P3M" in ann_file:
            self.dataset_name = "p3m"
        elif "OCHUMAN" in ann_file:
            self.dataset_name = "ochuman"
        else:
            # raise ValueError(f'Unknown dataset name in {ann_file}')
            self.dataset_name = ann_file.split("/")[-1].split(".")[0].split("_")[0]

    def get_image_path(self, image_id, dataset_name):
        if dataset_name == "refclef":
            return f"saiapr_tc-12/{str(image_id // 1000).zfill(2)}/images/{str(image_id)}.jpg"
        elif (
            dataset_name == "refcoco"
            or dataset_name == "refcoco+"
            or dataset_name == "refcocog"
        ):
            return f"mscoco/images/train2014/COCO_train2014_000000{str(image_id).zfill(6)}.jpg"
        else:
            raise ValueError(f"Unknown dataset name {dataset_name}")

    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file))
        ann_data = ann_list["data"]
        self.set_dataset_name(ann_file)
        print(f"loading {ann_file}..., dataset name: {self.dataset_name}")
        for img_info in ann_data:
            # 必须保证image_name是连续的
            image_name = img_info["image_name"]
            if isinstance(image_name, str):
                if ".png" in image_name:
                    continue
            if isinstance(image_name, str):
                if "." not in image_name:
                    image_name += ".jpg"
            else:
                image_name = self.get_image_path(image_name, self.dataset_name)
            img_path = os.path.join(self.root_path, image_name)
            height = img_info["height"]
            width = img_info["width"]
            gt_info = img_info["gt_ann"]
            captions = gt_info["caption"]
            if captions[0] == "Unknown":
                captions = [image_name[:-6]]
            if not isinstance(captions, list):
                caption = [captions]
            else:
                caption = captions
            for i in range(len(caption)):
                gt_mask = gt_info["segmentation"]

                qa_s = []
                question = (
                    DETAILED_QUESTIONS[0]
                    if self.args.use_mask_module
                    else DETAILED_QUESTIONS_NO_MASK[0]
                )
                if self.args.finish_action:
                    question = DETAILED_QUESTIONS_NO_MASK_FINISH[0]
                if self.args.predict_mode == "refine":
                    question = REFINE_PROMPT[0]
                question = question.replace("<region>", "<mask><pos>")
                # question = question.replace('<description>', caption)
                qa_s.append({"from": "human", "value": self.begin_str + question})
                data_infos.append(
                    dict(
                        img_path=img_path,
                        masks=None,
                        height=height,
                        width=width,
                        qas=qa_s,
                        caption=[caption[i]],
                        coor=None,
                        relative_coor=None,
                        is_positive=None,
                        click_id=0,
                        gt_mask=gt_mask,
                    )
                )

        return data_infos

    def annToMask(self, mask_ann, h, w):
        if mask_ann is None:
            return np.zeros((h, w), dtype=np.uint8)

        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann["counts"], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, h, w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def forward(
        self,
        root_path,
        ann_file,
    ):
        # initialize_distributed()
        self.captions_all = []
        self.gt_all = {}
        self.gt_all["images"] = []
        self.gt_all["annotations"] = []
        self.root_path = root_path
        self.coco = self.load_annotations(ann_file)
        self.ann_file = ann_file
        max_length = 5000
        id = 0
        trackers = {
            "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
            "union": AverageMeter("Union", ":6.3f", Summary.SUM),
            "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM),
        }
        previous_image_name = ""
        # record per image result to a csv
        import csv

        with open(
            f"{self.workspace}/{self.dataset_name}_{self.args.n_clicks}_{self.args.seg_model}_{self.args.grounding_model}_radius{self.args.undo_radius}_gridsample{self.args.use_grid_sample}_use_pmask_{self.args.use_previous_mask}_{self.visualize}use_gt_box{self.args.use_gt_box}.csv",
            "w",
        ) as fw:
            writer = csv.writer(fw)
            writer.writerow(["image_name", "mask_idx", "iou", "caption"])

            for data in tqdm(self.coco[:max_length]):
                # reform the data to fit the forward_single function
                
                if data["img_path"] != previous_image_name:
                    # a new image
                    if previous_image_name != "":
                        # compute the and update IOU
                        gt_mask_list = torch.stack(gt_mask_list).squeeze(1)
                        pred_mask_list = torch.stack(pred_mask_list).squeeze(1)

                        intersection, union, accuracy_iou = 0.0, 0.0, 0.0
                        for target, prediction, caption_, mask_idx in zip(
                            gt_mask_list,
                            pred_mask_list,
                            caption_list,
                            range(len(gt_mask_list)),
                        ):
                            intersect, union_, _ = intersectionAndUnionGPU(
                                prediction.contiguous().clone().float(),
                                target.contiguous().float(),
                                2,
                                ignore_index=255,
                            )
                            intersection += intersect
                            union += union_
                            accuracy_iou += intersect / (union_ + 1e-5)
                            # handles no-object targets
                            accuracy_iou[union_ == 0] += 1.0
                            writer.writerow(
                                [
                                    previous_image_name.split("/")[-1],
                                    mask_idx,
                                    (intersect / (union_ + 1e-5))[-1].item(),
                                    caption_,
                                ]
                            )
                        intersection, union = (
                            intersection.cpu().numpy(),
                            union.cpu().numpy(),
                        )
                        accuracy_iou = accuracy_iou.cpu().numpy() / len(gt_mask_list)
                        writer.writerow(
                            [
                                previous_image_name.split("/")[-1],
                                -1,
                                accuracy_iou[-1].item(),
                                "",
                            ]
                        )
                        trackers["intersection"].update(intersection)
                        trackers["union"].update(union)
                        trackers["gIoU"].update(accuracy_iou, n=len(gt_mask_list))
                        if self.visualize:
                            self.dataset_name = self.ann_file.split("/")[-1].split(".")[
                                0
                            ]
                            img_name = previous_image_name.split("/")[-1]

                            ori_image_path = previous_image_name
                            ori_image = cv2.imread(ori_image_path)
                            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
                            for i in range(len(pred_mask_list)):
                                path = f"{self.workspace}/visualize_{self.args.grounding_model}_radius{self.args.undo_radius}_{self.args.use_grid_sample}_{self.args.seg_model}_{self.args.vis_suffix}/segact/{self.dataset_name}_final/{img_name.replace('.', f'_caption{i}_pred.')}"
                                pred_mask = pred_mask_list[i]
                                gt_mask = gt_mask_list[i]
                                caption = caption_list[i]
                                visualize_mask_and_point(
                                    ori_image, pred_mask, None, path, True, caption
                                )
                                visualize_mask_and_point(
                                    ori_image,
                                    gt_mask,
                                    None,
                                    path.replace("pred", "gt"),
                                    True,
                                    caption,
                                )
                                txt_path = (
                                    path.replace(".jpg", ".txt")
                                    .replace(".png", ".txt")
                                    .replace(".jpeg", ".txt")
                                )
                                with open(txt_path, "w") as f:
                                    f.write("caption: " + caption)
                                    f.write("iou: " + str(accuracy_iou))

                    pred_mask_list = []
                    gt_mask_list = []
                    caption_list = []
                    previous_image_name = data["img_path"]

                data["annotation"] = [
                    {
                        "segmentation": data["masks"],
                        "id": id,
                        "click_id": data["click_id"],
                    }
                ]
                pred_mask = self.forward_single(data)
                if isinstance(pred_mask, tuple):
                    pred_mask, sample_record = pred_mask
                    if len(sample_record["pred_list"]) > 0:
                        pass
                    else:
                        print("pred_list is 0")
                    self.captions_all.append(sample_record)

                gt_mask = self.annToMask(data["gt_mask"], data["height"], data["width"])
                gt_mask = torch.from_numpy(gt_mask).unsqueeze(
                    0
                )  # DEBUG: convert_mask_to_coco_format(gt_mask[0])
                gt_mask_list.append(gt_mask)
                pred_mask_list.append(pred_mask)
                caption_list.append(data["caption"][0])

                id += 1

            # record the last image
            gt_mask_list = torch.stack(gt_mask_list).squeeze(1)
            pred_mask_list = torch.stack(pred_mask_list).squeeze(1)

            intersection, union, accuracy_iou = 0.0, 0.0, 0.0
            for target, prediction, caption_, mask_idx in zip(
                gt_mask_list, pred_mask_list, caption_list, range(len(gt_mask_list))
            ):
                intersect, union_, _ = intersectionAndUnionGPU(
                    prediction.contiguous().clone().float(),
                    target.contiguous().float(),
                    2,
                    ignore_index=255,
                )
                intersection += intersect
                union += union_
                accuracy_iou += intersect / (union_ + 1e-5)
                # handles no-object targets
                accuracy_iou[union_ == 0] += 1.0
                writer.writerow(
                    [
                        previous_image_name.split("/")[-1],
                        mask_idx,
                        (intersect / (union_ + 1e-5))[-1].item(),
                        caption_,
                    ]
                )
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            accuracy_iou = accuracy_iou.cpu().numpy() / len(gt_mask_list)
            writer.writerow(
                [previous_image_name.split("/")[-1], -1, accuracy_iou[-1].item(), ""]
            )
            trackers["intersection"].update(intersection)
            trackers["union"].update(union)
            trackers["gIoU"].update(accuracy_iou, n=len(gt_mask_list))
            if self.visualize:
                self.dataset_name = self.ann_file.split("/")[-1].split(".")[0]
                img_name = previous_image_name.split("/")[-1]

                ori_image_path = previous_image_name
                ori_image = cv2.imread(ori_image_path)
                ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
                for i in range(len(pred_mask_list)):
                    path = f"{self.workspace}/visualize_{self.args.grounding_model}_radius{self.args.undo_radius}_{self.args.use_grid_sample}_{self.args.seg_model}_{self.args.vis_suffix}/segact/{self.dataset_name}_final/{img_name.replace('.', f'_caption{i}_pred.')}"
                    pred_mask = pred_mask_list[i]
                    gt_mask = gt_mask_list[i]
                    caption = caption_list[i]
                    visualize_mask_and_point(
                        ori_image, pred_mask, None, path, True, caption
                    )
                    visualize_mask_and_point(
                        ori_image,
                        gt_mask,
                        None,
                        path.replace("pred", "gt"),
                        True,
                        caption,
                    )
                    txt_path = (
                        path.replace(".jpg", ".txt")
                        .replace(".png", ".txt")
                        .replace(".jpeg", ".txt")
                    )
                    with open(txt_path, "w") as f:
                        f.write("caption: " + caption)
                        f.write("iou: " + str(accuracy_iou))

        # save record
        self.dataset_name = self.ann_file.split("/")[-1].split(".")[0]
        json_save_path = f"{self.workspace}/{self.dataset_name}_newresults_{self.args.n_clicks}_{self.args.seg_model}_{self.args.grounding_model}_radius{self.args.undo_radius}use_gt_box{self.args.use_gt_box}.json"
        with open(json_save_path, "w") as f:
            json.dump(self.captions_all, f)

        for meter in trackers.values():
            meter.all_reduce()

        iou_per_class = trackers["intersection"].sum / (trackers["union"].sum + 1e-10)
        class_iou = iou_per_class[1]
        global_iou = trackers["gIoU"].avg[1]

        print(class_iou, global_iou)
        self.dataset_name = self.ann_file.split("/")[-1].split(".")[0]
        self.segmodel_name = self.args.checkpoint.split("/")[-1].split(".")[0]
        with open(
            f"{self.workspace}/{self.dataset_name}_{self.args.n_clicks}_{self.segmodel_name}_{self.args.seg_model}_{self.args.grounding_model}_radius{self.args.undo_radius}_gridsample{self.args.use_grid_sample}_use_pmask_{self.args.use_previous_mask}_use_gt_box{self.args.use_gt_box}.txt",
            "w",
        ) as f:
            f.write(f"class_iou: {class_iou}, global_iou: {global_iou}")

    def image_process_from_simpleclick(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def update_pred_list(
        self, inputs, click_id, gt_box, previous_mask, clicks, outputs=None
    ):
        click_info_dict = {}
        click_info_dict["click_id"] = click_id
        click_info_dict["clicks"] = clicks
        click_info_dict["used_box"] = gt_box
        click_info_dict["mask"] = convert_mask_to_coco_format(previous_mask)
        click_info_dict["outputs"] = outputs
        inputs["pred_list"].append(click_info_dict)
        return inputs

    def forward_single(self, inputs):
        img_path = inputs["img_path"]
        height = inputs["height"]
        width = inputs["width"]
        annotations = inputs["annotation"]
        gt_mask = inputs["gt_mask"]
        # Initialize models
        segmentation_model = self.segmodel
        grounding_model = self.groundingmodel
        last_source = dict()
        inputs["pred_list"] = []
        # Process image
        simple_click_image = segmentation_model.image_process(img_path=img_path)

        with torch.no_grad():
            segmentation_model.set_input_image(simple_click_image)

            clicker = Clicker()
            previous_mask = None
            points = None
            pred_logits = None

            for i in range(self.args.n_clicks):
                ann = annotations[0]
                id = ann["id"]
                if i == 0:
                    mask_r = ann["segmentation"]
                    mask = self.annToMask(mask_r, height, width)
                    mask = torch.from_numpy(mask).unsqueeze(0)  # 1xHxW, uint8
                    click_id = ann["click_id"]
                    last_pred_logits = None
                    gt_box2 = None
                    last_ref_box_str = None
                    if self.args.only_use_gt_box or self.args.use_gt_box:
                        gt_box = maskUtils.toBbox(gt_mask)  # x,y,w,h
                        gt_box = [
                            gt_box[0],
                            gt_box[1],
                            gt_box[0] + gt_box[2],
                            gt_box[1] + gt_box[3],
                        ]  # x1,y1,x2,y2
                        pred_mask = segmentation_model.get_prediction(
                            clicker=None, box=gt_box
                        )
                        if isinstance(pred_mask, tuple):
                            pred_mask, pred_logits = pred_mask
                        previous_mask = (
                            torch.from_numpy(pred_mask).to(torch.uint8).unsqueeze(0)
                        )
                        inputs = self.update_pred_list(
                            inputs, click_id, gt_box, previous_mask, []
                        )
                        if self.args.only_use_gt_box:
                            return previous_mask, inputs
                        else:
                            gt_box2 = gt_box
                            if self.args.use_previous_mask:
                                last_pred_logits = pred_logits
                            continue
                else:
                    mask = previous_mask
                    click_id = i


                prompt, conv = grounding_model.build_prompt(
                    inputs, last_ref_box_str
                )
                outputs = grounding_model.generate_response(
                    prompt, img_path, mask, conv
                )
                if last_ref_box_str is not None:
                    outputs = last_ref_box_str + outputs
                print(f"outputs: {outputs}")
                if self.args.use_last_box:
                    # '<ref>person bottom left</ref><box>(68,308),(255,986)</box> Positive point: (679, 186)'
                    last_box_str = outputs.split("<box>")[-1].split("</box>")[0]
                    last_ref_str = outputs.split("<ref>")[-1].split("</ref>")[0]
                    last_ref_box_str = (
                        f"<ref>{last_ref_str}</ref><box>{last_box_str}</box>"
                    )

                inputs["outputs"] = outputs

                if "box" in self.args.grounding_model:
                    is_positive, points, gt_box = grounding_model.process_response(
                        outputs
                    )
                    scale = 999
                    # gt_box = [gt_box[0]/999, gt_box[1]/999, gt_box[2]/999, gt_box[3]/999]'
                    gt_box = [
                        gt_box[0] / scale,
                        gt_box[1] / scale,
                        gt_box[2] / scale,
                        gt_box[3] / scale,
                    ]
                    # convert relative coor to absolute coor
                    gt_box2 = [
                        int(gt_box[0] * width),
                        int(gt_box[1] * height),
                        int(gt_box[2] * width),
                        int(gt_box[3] * height),
                    ]
                else:
                    is_positive, points = grounding_model.process_response(outputs)


                if is_positive is None:
                    # Handle finish action or invalid output
                    if self.visualize:
                        self.dataset_name = self.ann_file.split("/")[-1].split(".")[0]
                        img_name = img_path.split("/")[-1]
                        path = f"{self.workspace}/visualize_{self.args.grounding_model}_radius{self.args.undo_radius}_{self.args.use_grid_sample}_{self.args.seg_model}_{self.args.vis_suffix}/segact/{self.dataset_name}_trace/{img_name.replace('.', f'_{id}_{click_id}.')}"
                        if not os.path.exists(os.path.dirname(path)):
                            os.makedirs(os.path.dirname(path))
                        if self.args.finish_action and "Finish" in outputs:
                            path = path.replace(".", "finish.")
                        else:
                            path = path.replace(".", "invalid.")
                        # image = torchshow.visualization.auto_unnormalize_image(image)
                        image = cv2.imread(img_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        txt_path = (
                            path.replace(".jpg", ".txt")
                            .replace(".png", ".txt")
                            .replace(".jpeg", ".txt")
                        )
                        with open(txt_path, "w") as f:
                            f.write("generated caption: " + outputs)
                            f.write("caption: " + inputs["caption"][0])
                            f.write("gt_coors: " + str(inputs["relative_coor"]))
                    if previous_mask is not None:
                        break
                    else:
                        previous_mask = mask
                        break


                # Update clicker and get new prediction
                abs_points = (round(points[0] * height), round(points[1] * width))

                click = Click(is_positive=is_positive, coords=abs_points)
                clicker.add_click(click, self.args.undo_radius)

                if self.args.use_previous_mask:
                    last_pred_logits = pred_logits
                pred_mask = segmentation_model.get_prediction(
                    clicker, box=gt_box2, mask=last_pred_logits
                )
                if isinstance(pred_mask, tuple):
                    pred_mask, pred_logits = pred_mask

                previous_mask = (
                    torch.from_numpy(pred_mask).to(torch.uint8).unsqueeze(0)
                )  # update the previous mask
                inputs = self.update_pred_list(
                    inputs,
                    click_id,
                    gt_box2,
                    previous_mask,
                    {"is_positive": is_positive, "coords": abs_points},
                    outputs,
                )
                # Visualization (optional)
                if self.visualize:
                    self.dataset_name = self.ann_file.split("/")[-1].split(".")[0]
                    img_name = img_path.split("/")[-1]
                    path = f"{self.workspace}/visualize_{self.args.grounding_model}_radius{self.args.undo_radius}_{self.args.use_grid_sample}_{self.args.seg_model}_{self.args.vis_suffix}/segact/{self.dataset_name}_trace/{img_name.replace('.', f'_{id}_{click_id}.')}"
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    visualize_mask_and_point(
                        image, previous_mask, points, path, is_positive, outputs
                    )
                    # try save point_list and box
                    try:
                        point_list = [
                            clicker.clicks_list[i].coords
                            for i in range(len(clicker.clicks_list))
                        ]
                        label_list = [
                            clicker.clicks_list[i].is_positive
                            for i in range(len(clicker.clicks_list))
                        ]
                        path = path.replace(".jpg", "_point.jpg")
                        visualize_mask_and_pointlist(
                            image,
                            previous_mask,
                            point_list,
                            path,
                            label_list,
                            outputs,
                            gt_box2,
                        )
                    except Exception as e:
                        pass
                    txt_path = (
                        path.replace(".jpg", ".txt")
                        .replace(".png", ".txt")
                        .replace(".jpeg", ".txt")
                    )
                    with open(txt_path, "w") as f:
                        f.write("generated caption: " + outputs)
                        f.write("caption: " + inputs["caption"][0])
                        f.write("gt_coors: " + str(inputs["relative_coor"]))

            return previous_mask, inputs
