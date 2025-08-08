# Example usage configurations

# Basic evaluation
python eval_result_iou.py --input_json ./results/refcoco+_val_predictions.json

# Evaluate all splits with specific GPU
python eval_result_iou.py \
    --input_json ./results/refcoco+_val_predictions.json \
    --eval_all_splits \
    --cuda_device 0 


# Evaluate specific click with visualization
python eval_result_iou.py \
    --input_json ./results/refcoco+_val_predictions.json \
    --click_id 2 \
    --visualize

# Batch evaluation script
for json_file in ./results/*.json; do
    echo "Evaluating $json_file"
    python eval_result_iou.py --input_json "$json_file"
done
