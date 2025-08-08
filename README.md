<div align="center">

# 🎯 SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories

[Muzhi Zhu](https://scholar.google.com/citations?user=064gBH4AAAAJ&hl=en)<sup>1,2</sup>, &nbsp;
Yuzhuo Tian<sup>1</sup>, &nbsp;
[Hao Chen](https://stan-haochen.github.io/)<sup>1*</sup>, &nbsp;
Chunluan Zhou<sup>2</sup>, &nbsp;
Qingpei Guo<sup>2*</sup>, &nbsp;
Yang Liu<sup>1</sup>, &nbsp;
Ming Yang<sup>2</sup>, &nbsp;
[Chunhua Shen](https://cshen.github.io/)<sup>1*</sup>

<sup>1</sup>[Zhejiang University](https://www.zju.edu.cn/english/), &nbsp;
<sup>2</sup>[Ant Group](https://www.antgroup.com/en)

**CVPR2025**

[📄 **Paper**](https://arxiv.org/abs/2503.08625)&nbsp; | &nbsp;[🌐 **Project Page**](https://aim-uofa.github.io/SegAgent/)&nbsp; | &nbsp;[🤖 **Model Weight**](https://www.modelscope.cn/models/zzzmmz/SegAgent-Model)&nbsp; | &nbsp;[📊 **Data**](https://www.modelscope.cn/models/zzzmmz/SegAgent-Dataset)
</div>

## 🚀 Overview
<div align="center">
<img width="800" alt="SegAgent Framework" src="images/framework.png">
</div>

## 📖 Description

Multimodal Large Language Models (MLLMs) demonstrate remarkable capabilities in understanding images but still struggle with pixel-level tasks like segmentation. SegAgent addresses this by introducing a novel **Human-Like Mask Annotation Task (HLMAT)**, enabling MLLMs to mimic the annotation trajectories of human experts using interactive segmentation tools.

SegAgent effectively leverages these annotation trajectories without requiring architectural modifications or additional implicit tokens. Our approach significantly enhances MLLMs' segmentation and mask refinement abilities, establishing a new paradigm for assessing fine-grained visual understanding and multi-step reasoning.







## 🚩 Plan
- ✅ Release the weights.
- ✅ Release the inference code.
- ✅ Release the trajectory data for training and evaluation.



## 🚀 Getting Started
```bash
pip install -r  env.txt
```

## 🤖 Inference

You can run inference on the validation or test set using the trained model and the provided script:

```bash
bash run_eval.sh /path/to/your/trained_model
```

This will run inference with **SimpleClick** as the segmentation model and **SegAgent** as the language grounding model. The script processes images and saves the predictions to the output directory.

To evaluate the results, run:

```bash
python eval_result_iou.py --input_json ./results/refcoco+_val_predictions.json
```

📄 For more details, refer to [`./evaltools/eval.md`](./evaltools/eval.md).

---

## 🧑‍🏫 Training

SegAgent is trained using **Human-Like Mask Annotation Trajectories (HLMAT)**. Follow the steps below to launch the training process:

### Step 1: Prepare the Data

Ensure that the annotation trajectory data is preprocessed and saved in the appropriate format (e.g., COCO-style JSON files + click sequences).

We have uploaded the preprocessed trajectory data here:  
📁 [SegAgent-Data](https://www.modelscope.cn/models/zzzmmz/SegAgent-Dataset)

Example structure:

```bash
tree ./data/segagent-data
├── refcoco_train.json
├── refcoco_val.json
├── refcoco+_train.json
├── ...
```

Additional image data sources:
- RefCOCO image datasets: [LISA GitHub Repository](https://github.com/dvlab-research/LISA)
- HQ segmentation (SAM-HQ): [Hugging Face SAM-HQ Data](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/data)

### Step 2: Run Training

We recommend converting the trajectory data into a format supported by [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and training using their framework directly.

---



## 🎫 License

For academic usage, this project is licensed under [the 2-clause BSD License](LICENSE). For commercial inquiries, please contact [Chunhua Shen](mailto:chhshen@gmail.com).

## 🖊️ Citation

If you find this work helpful for your research, please cite:

```BibTeX
@article{zhu2025segagent,
  title={SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories},
  author={Zhu, Muzhi and Tian, Yuzhuo and Chen, Hao and Zhou, Chunluan and Guo, Qingpei and Liu, Yang and Yang, Ming and Shen, Chunhua},
  journal={arXiv preprint arXiv:2503.08625},
  year={2025},
  url={https://arxiv.org/abs/2503.08625}
}