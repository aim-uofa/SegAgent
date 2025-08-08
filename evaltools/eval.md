# 🧪 Segmentation Evaluation with SegAgent




## 🚀 Getting Started

### 1. Requirements

- Linux environment with Bash
- GPU with CUDA support
- Python environment compatible with `main.py`
- Necessary models and datasets downloaded:
  - SimpleClick checkpoint
  - Pretrained vision-language model (e.g., Qwen)
  - RefCOCO dataset in JSON format and image files

---

### 2. Setup

1. **Clone the repository (or place this script in your working directory):**

   ```bash
   cd /home/zmz/code/SegAgent/evaltools
   ```
2. **Ensure you have the required models and datasets:**

    Checkpoint could be downloaded from [here](https://www.modelscope.cn/models/zzzmmz/SegAgent-Model)

    Json dataset files can be downloaded from [here](https://www.modelscope.cn/models/zzzmmz/SegAgent-Dataset)

3. **Verify the dataset and checkpoint paths are correctly configured in `run_eval.sh`.**

---

### 3. Run the Script

```bash
bash run_eval.sh /absolute/path/to/your/model
```


> The script supports changing the number of clicks (default is 7). You can modify the loop in the script to sweep over multiple values, such as `{1..5}`.

---

### 4. Example Usage for Results Evaluation

After running the evaluation script, you can use the `example_usage.sh` file as a reference for evaluating your results with different configurations.

#### Basic Usage Examples:

1. **Basic evaluation:**
   ```bash
   python eval_result_iou.py --input_json ./results/refcoco+_val_predictions.json
   ```

2. **Evaluate all splits with specific GPU:**
   ```bash
   python eval_result_iou.py \
       --input_json /path/to/your/results.json \
       --eval_all_splits \
       --cuda_device 0
   ```

3. **Evaluate specific click with visualization:**
   ```bash
   python eval_result_iou.py \
       --input_json ./results/refcoco+_val_predictions.json \
       --click_id 2 \
       --visualize
   ```

4. **Batch evaluation for multiple result files:**
   ```bash
   for json_file in ./results/*.json; do
       echo "Evaluating $json_file"
       python eval_result_iou.py --input_json "$json_file"
   done
   ```

> **Note:** You can find more detailed usage examples in `example_usage.sh`. Modify the paths and parameters according to your specific setup and requirements.

---

## ⚙️ Script Configuration

The script runs `main.py` with the following options:

- Model path from input
- Image directory: `data/refer_seg/images`
- Dataset JSON: `refcoco_testA.json`
- Segmentation model: `simple_click`
- Grounding model: `qwen-full`
- SimpleClick checkpoint: `cocolvis_vit_large.pth`

You can change these values inside `run_eval.sh` as needed.

---



## 📬 Feedback

If you encounter issues or want to contribute improvements, feel free to open an issue or PR.

---

