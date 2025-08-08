# merge json list into one json file
# data/record_trace/LVIS_v1_train_0_30000.json
# data/record_trace/LVIS_v1_train_30000_60000.json
# data/record_trace/LVIS_v1_train_60000_-1.json
json_list = [
    'data/record_trace/LVIS_v1_train_0_30000.json',
    'data/record_trace/LVIS_v1_train_30000_60000.json',
    'data/record_trace/LVIS_v1_train_60000_-1.json'
]
import json
data = []
info = None
for json_path in json_list:
    with open(json_path, 'r') as f:
        temp = json.load(f)
    print(json_path, len(temp['data']))
    data += temp['data']
    if info is None:
        info = temp['info']
    
new_json = { 'info': info, 'data': data }
new_json_path = 'data/record_trace/LVIS_v1_train.json'
with open(new_json_path, 'w') as f:
    json.dump(new_json, f)
