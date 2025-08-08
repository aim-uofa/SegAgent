## 数据集注册
新增 isegm/data/datasets/ref_rea_seg.py(注意要保存caption)
修改 isegm/inference/utils.py

目前支持的新数据集
isegm/data/datasets/lvis_v1.py
isegm/data/datasets/pascalvoc.py
isegm/data/datasets/ref_rea_seg.py
## 使用脚本生成轨迹
run_gen_refsegtrain_clef.sh
run_gen_refsegtrain_coco.sh
run_gen_refsegtrain_coco+.sh
run_gen_refsegtrain_cocog.sh


## 验证生成的轨迹

tools/check_record.py