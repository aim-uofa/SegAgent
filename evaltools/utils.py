from enum import Enum
from unittest.mock import MagicMock
import numpy as np
import torch
import torch.distributed as dist
import os
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_UNK_TOKEN = '<unk>'

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_BBOX_TOKEN = "<bbox>"


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


def initialize_distributed():
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    
    # 将 MASTER_PORT 设置为 12345 + CUDA_VISIBLE_DEVICES 的值
    os.environ['MASTER_PORT'] = str(12345 + int(cuda_visible_devices))
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif isinstance(v, list) and len(v) > 0:
            input_dict[k] = [ele.cuda(non_blocking=True) if isinstance(ele, torch.Tensor) else ele for ele in v]
    return input_dict

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target



def setup_model_and_data():
    # 创建一个模拟的模型
    model = MagicMock()
    model.eval = MagicMock()
    model.return_value = {
        "pred_masks": [torch.tensor([[[0.1, 0.9], [0.8, 0.2]]])],
        "gt_masks": [torch.tensor([[[0, 1], [1, 0]]])]
    }

    # 创建一个模拟的数据加载器
    validation_loader = [MagicMock() for _ in range(2)]
    for batch in validation_loader:
        batch.return_value = {
            "global_enc_images": torch.randn(1, 3, 224, 224),
            "grounding_enc_images": torch.randn(1, 3, 224, 224)
        }

    args = MagicMock()
    args.local_rank = 0

    return model, validation_loader, args

def test_infer_and_evaluate(setup_model_and_data):
    initialize_distributed()
    model, validation_loader, args = setup_model_and_data()
    
    trackers = {
        "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
        "union": AverageMeter("Union", ":6.3f", Summary.SUM),
        "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
    }

    model.eval()
    for data_batch in validation_loader:
        # Prepare data and convert relevant tensors to the appropriate type
        data_batch = dict_to_cuda(data_batch)
        for key in ["global_enc_images", "grounding_enc_images"]:
            data_batch[key] = data_batch[key].to(dtype=torch.bfloat16, device=args.local_rank)

        torch.cuda.empty_cache()

        # Model inference without gradient tracking
        with torch.no_grad():
            results = model(**data_batch)

        predictions = results["pred_masks"]
        gt_masks = results["gt_masks"][0].int()
        predicted_masks = (predictions[0] > 0).int()  # Thresholding to get binary masks
        assert len(predictions) == 1

        intersection, union, accuracy_iou = 0.0, 0.0, 0.0
        for target, prediction in zip(gt_masks, predicted_masks):
            intersect, union_, _ = intersectionAndUnionGPU(
                prediction.contiguous().clone().float(), target.contiguous().float(), 2, ignore_index=255
            )
            intersection += intersect
            union += union_
            accuracy_iou += intersect / (union_ + 1e-5)
            # handles no-object targets
            accuracy_iou[union_ == 0] += 1.0

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        accuracy_iou = accuracy_iou.cpu().numpy() / gt_masks.shape[0]
        trackers["intersection"].update(intersection)
        trackers["union"].update(union)
        trackers["gIoU"].update(accuracy_iou, n=gt_masks.shape[0])

    for meter in trackers.values():
        meter.all_reduce()

    iou_per_class = trackers["intersection"].sum / (trackers["union"].sum + 1e-10)
    class_iou = iou_per_class[1]
    global_iou = trackers["gIoU"].avg[1]

    assert isinstance(global_iou, float)
    assert isinstance(class_iou, float)
    assert 0 <= global_iou <= 1
    assert 0 <= class_iou <= 1

#test_infer_and_evaluate(setup_model_and_data)