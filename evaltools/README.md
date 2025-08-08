# IoU Evaluation Tool

This tool evaluates Intersection over Union (IoU) metrics for segmentation results on RefCOCO family datasets.

## Features

- Evaluate segmentation results on RefCOCO, RefCOCO+, and RefCOCOg datasets
- Support for multiple evaluation splits (val, testA, testB, test)
- Flexible click-based evaluation
- CSV output with detailed per-image results
- Optional visualization support

## Requirements

- Python 3.7+
- PyTorch
- pycocotools
- tqdm
- numpy

## Installation

```bash
# Install required packages
pip install torch torchvision pycocotools tqdm numpy
```

## Usage

### Basic Usage

Evaluate a single prediction file:

```bash
python eval_result_iou.py --input_json /path/to/predictions.json
```

### Advanced Usage

```bash
# Evaluate with specific CUDA device
python eval_result_iou.py --input_json /path/to/predictions.json --cuda_device 0

# Evaluate all RefCOCO family splits
python eval_result_iou.py --input_json /path/to/refcoco+_val_predictions.json --eval_all_splits

# Evaluate specific click ID
python eval_result_iou.py --input_json /path/to/predictions.json --click_id 2

# Enable visualization output
python eval_result_iou.py --input_json /path/to/predictions.json --visualize
```

## Arguments

- `--input_json`: Path to input JSON file with predictions (required)
- `--cuda_device`: CUDA device to use (default: '0')
- `--click_id`: Specific click ID to evaluate (default: None, uses best IoU)
- `--visualize`: Save visualization results (flag)
- `--eval_all_splits`: Evaluate all RefCOCO family splits (flag)

## Input Format

The input JSON file should contain a list of prediction data with the following structure:

```json
[
  {
    "img_path": "/path/to/image.jpg",
    "height": 480,
    "width": 640,
    "caption": ["object description"],
    "gt_mask": {
      "counts": "compressed_rle_string",
      "size": [480, 640]
    },
    "pred_list": [
      {
        "click_id": 0,
        "mask": {
          "counts": "compressed_rle_string",
          "size": [480, 640]
        },
        "outputs": "Current IOU: 85.2, Positive point: (0.5, 0.3)"
      }
    ],
    "click_id": 0
  }
]
```

## Output

The tool generates several output files:

1. **CSV file** (`*_clicknewNone.csv`): Detailed per-image IoU results
2. **Text file** (`*_clickNone.txt`): Summary metrics (class IoU and global IoU)

### CSV Format

| image_name | mask_idx | iou | caption |
|------------|----------|-----|---------|
| image1.jpg | 0 | 0.856 | object description |
| image1.jpg | -1 | 0.856 | (summary for image) |

## Evaluation Splits

When using `--eval_all_splits`, the tool evaluates the following standard splits:

- RefCOCOg: val, test (UMD split)
- RefCOCO: val, testA, testB (UNC split)  
- RefCOCO+: val, testA, testB (UNC split)

## Error Handling

- The tool gracefully handles missing files and skips them
- Existing output files are skipped to avoid recomputation
- Robust parsing of IoU values from model outputs

## License

This code is released under the same license as the parent project.
