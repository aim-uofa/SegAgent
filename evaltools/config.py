import argparse

def get_config():
    parser = argparse.ArgumentParser(
        description="osprey demo", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--model",
        help="path to osprey model",
        default="path/to/Osprey-7B-refcocog-fintune",
    )
    parser.add_argument(
        "--img", help="path to coco imgs", default="path/to/coco_all_imgs/"
    )
    parser.add_argument(
        "--json",
        help="path to refcocog val json file",
        default="./finetune_refcocog_val_with_mask.json",
    )
    parser.add_argument(
        "--use_mask_module",
        dest="use_mask_module",
        action="store_true",
        help="use mask module",
    )
    parser.add_argument(
        "--no_use_mask_module",
        dest="use_mask_module",
        action="store_false",
        help="do not use mask module",
    )
    parser.set_defaults(use_mask_module=True)
    # simpleclick args
    parser.add_argument(
        "mode",
        choices=["NoBRS", "RGB-BRS", "DistMap-BRS", "f-BRS-A", "f-BRS-B", "f-BRS-C"],
        help="",
    )

    group_checkpoints = parser.add_mutually_exclusive_group(required=True)
    group_checkpoints.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="The path to the checkpoint. "
        "This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) "
        "or an absolute path. The file extension can be omitted.",
    )
    group_checkpoints.add_argument(
        "--exp-path",
        type=str,
        default="",
        help="The relative path to the experiment with checkpoints."
        "(relative to cfg.EXPS_PATH)",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default="GrabCut,Berkeley,DAVIS,PascalVOC,SBD,BraTS,ssTEM,OAIZIB,COCO_MVal",
        help="List of datasets on which the model should be tested. "
        "Datasets are separated by a comma. Possible choices: "
        "GrabCut, Berkeley, DAVIS, SBD, PascalVOC",
    )

    group_device = parser.add_mutually_exclusive_group()
    group_device.add_argument("--gpus", type=str, default="0", help="ID of used GPU.")
    group_device.add_argument(
        "--cpu", action="store_true", default=False, help="Use only CPU for inference."
    )
    group_iou_thresh = parser.add_mutually_exclusive_group()
    group_iou_thresh.add_argument(
        "--target-iou",
        type=float,
        default=0.98,
        help="Target IoU threshold for the NoC metric. (min possible value = 0.8)",
    )
    group_iou_thresh.add_argument(
        "--iou-analysis",
        action="store_true",
        default=False,
        help="Plot mIoU(number of clicks) with target_iou=1.0.",
    )

    parser.add_argument(
        "--n-clicks",
        type=int,
        default=5,
        help="Maximum number of clicks for the NoC metric.",
    )
    parser.add_argument(
        "--min-n-clicks",
        type=int,
        default=1,
        help="Minimum number of clicks for the evaluation.",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        required=False,
        default=0.49,
        help="The segmentation mask is obtained from the probability outputs using this threshold.",
    )
    parser.add_argument("--clicks-limit", type=int, default=None)
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="cvpr",
        help="Possible choices: cvpr, fixed<number>, or fixed<number>,<number>,(e.g. fixed400, fixed400,600).",
    )
    parser.add_argument("--eval-ritm", action="store_true", default=False)
    parser.add_argument("--save-ious", action="store_true", default=False)
    parser.add_argument("--print-ious", action="store_true", default=False)
    parser.add_argument("--vis-preds", action="store_true", default=False)
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="The model name that is used for making plots.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="../third_party/SimpleClick/config.yml",
        help="The path to the config file.",
    )
    parser.add_argument(
        "--logs-path",
        type=str,
        default="",
        help="The path to the evaluation logs. Default path: cfg.EXPS_PATH/evaluation_logs.",
    )
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--record_trace", action="store_true", default=False)
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="The split of the dataset that should be used for evaluation.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--coord_type",
        type=str,
        default="float",
        help="The type of the coordinates. Possible choices: float, int.",
    )
    parser.add_argument(
        "--mask_color",
        type=str,
        default="green",
        help="The color of the mask. Possible choices: green, red, blue",
    )
    parser.add_argument("--use_mask_overlay", action="store_true", default=False)
    parser.add_argument("--finish_action", type=int, default=0)
    parser.add_argument(
        "--seg_model",
        choices=["simple_click", "sam", "sam_l", "sam_h"],
        default="simple_click",
    )
    parser.add_argument(
        "--grounding_model",
        choices=[

            "qwen-full",

        ],
    )
    parser.add_argument("--only_use_gt_box", type=int, default=0)
    parser.add_argument("--use_gt_box", type=int, default=0)
    parser.add_argument("--use_grid_sample", type=int, default=0)
    parser.add_argument("--use_previous_mask", type=int, default=0)
    parser.add_argument("--predict_mode", type=str, default="together")
    parser.add_argument("--use_last_box", type=int, default=0)
    parser.add_argument("--vis_suffix", type=str, default="")
    parser.add_argument("--undo_radius", type=int, default=0)
    args = parser.parse_args()
    return args
