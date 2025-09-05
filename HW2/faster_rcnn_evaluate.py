import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
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
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
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
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        img_tensor = transform(image).to(device)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            predictions = model([img_tensor])
        end_time = time.time()

        inference_times.append(end_time - start_time)
        mem_usages.append(psutil.Process().memory_info().rss / (1024 * 1024)) # in MB

        # Format results
        for i in range(len(predictions[0]['labels'])):
            bbox = predictions[0]['boxes'][i].cpu().numpy()
            results.append({
                'image_id': img_id,
                'category_id': predictions[0]['labels'][i].item(),
                'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])],
                'score': float(predictions[0]['scores'][i].item()),
            })

    # Save results
    with open('faster_rcnn_results.json', 'w') as f:
        json.dump(results, f)

    # Evaluate
    coco_dt = coco.loadRes('faster_rcnn_results.json')
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Performance Metrics
    print(f"Average Inference Time: {np.mean(inference_times) * 1000:.2f} ms")
    print(f"Average Memory Usage: {np.mean(mem_usages):.2f} MB")

if __name__ == '__main__':
    main()
