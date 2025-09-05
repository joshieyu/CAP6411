
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
ann_file = os.path.join(script_dir, 'cocoDataset', 'annotations', 'instances_val2017.json')

with open(ann_file, 'r') as f:
    data = json.load(f)

categories = data['categories']

# Print in a format that can be pasted into a Python script
print("COCO_CATEGORIES = [")
for cat in categories:
    print(f"    {{\"id\": {cat['id']}, \"name\": \"{cat['name']}\", \"supercategory\": \"{cat['supercategory']}\"}},")
print("]")
