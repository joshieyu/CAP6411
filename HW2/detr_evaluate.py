import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import numpy as np
import time
import psutil
import json
import os

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = script_dir
    ann_file = os.path.join(data_dir, 'cocoDataset', 'annotations', 'instances_val2017.json')
    img_dir = os.path.join(data_dir, 'cocoDataset', 'val2017')

    # Load COCO
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    # Load Model
    processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    results = []
    inference_times = []
    mem_usages = []

    for i, img_id in enumerate(img_ids):
        if (i + 1) % 100 == 0:
            print(f"Processing image {i + 1}/{len(img_ids)}")
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # Preprocess
        inputs = processor(images=image, return_tensors='pt').to(device)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()

        inference_times.append(end_time - start_time)
        mem_usages.append(psutil.Process().memory_info().rss / (1024 * 1024)) # in MB

        # Format results
        target_sizes = torch.tensor([image.size[::-1]])
        processed_outputs = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

        for score, label, box in zip(processed_outputs["scores"], processed_outputs["labels"], processed_outputs["boxes"]):
            results.append({
                'image_id': img_id,
                'category_id': label.item(),
                'bbox': [float(box[0].item()), float(box[1].item()), float((box[2] - box[0]).item()), float((box[3] - box[1]).item())],
                'score': float(score.item()),
            })

    # Save results
    with open('detr_results.json', 'w') as f:
        json.dump(results, f)

    # Evaluate
    coco_dt = coco.loadRes('detr_results.json')
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Performance Metrics
    print(f"Average Inference Time: {np.mean(inference_times) * 1000:.2f} ms")
    print(f"Average Memory Usage: {np.mean(mem_usages):.2f} MB")

if __name__ == '__main__':
    main()
