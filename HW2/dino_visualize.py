
import os
import sys
import random
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Add DINO project to python path
script_dir = os.path.dirname(os.path.abspath(__file__))
dino_dir = os.path.join(script_dir, 'DINO')
sys.path.append(dino_dir)

from util.slconfig import SLConfig
from models.registry import MODULE_BUILD_FUNCS
import datasets.transforms as T
from util.misc import nested_tensor_from_tensor_list
from coco_categories import COCO_CATEGORIES


def build_dino_model(args):
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, postprocessors

def get_coco_categories():
    return {cat['id']: cat['name'] for cat in COCO_CATEGORIES}

def visualize_predictions(image, predictions, category_mapping, threshold=0.5):
    draw = ImageDraw.Draw(image)
    
    for score, label, box in zip(predictions["scores"], predictions["labels"], predictions["boxes"]):
        if score > threshold:
            box = [round(i, 2) for i in box.tolist()]
            label_id = label.item()
            label_name = category_mapping.get(label_id, "N/A")
            
            print(
                f"Detected {label_name} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
            
            draw.rectangle(box, outline="red", width=3)
            
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            
            text = f"{label_name}: {round(score.item(), 2)}"
            text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
            draw.rectangle(text_bbox, fill="red")
            draw.text((box[0], box[1]), text, fill="white", font=font)
            
    return image

def main():
    # --- Config --- #
    config_file = os.path.join(dino_dir, 'config/DINO/DINO_4scale.py')
    checkpoint_path = os.path.join(dino_dir, 'checkpoint0011_4scale.pth')
    img_dir = os.path.join(script_dir, 'cocoDataset', 'val2017')
    output_dir = os.path.join(script_dir, 'inference_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Load Config --- #
    args = SLConfig.fromfile(config_file)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.resume = checkpoint_path
    args.coco_path = os.path.join(script_dir, 'cocoDataset') # Not used, but good to have
    print(f"Using device: {args.device}")

    # --- Build Model --- #
    model, postprocessors = build_dino_model(args)
    model.to(args.device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # --- Get category mapping ---
    category_mapping = get_coco_categories()

    while True:
        # --- Image Selection and Preprocessing --- #
        image_files = os.listdir(img_dir)
        random_image_file = random.choice(image_files)
        img_path = os.path.join(img_dir, random_image_file)
        print(f"\nProcessing image: {img_path}")
        image = Image.open(img_path).convert('RGB')

        # --- Transformations --- # 
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # The default scales and max_size are used here as in the original implementation
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        max_size = 1333
        transform = T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])
        image_transformed, _ = transform(image, None)

        # --- Inference --- #
        with torch.no_grad():
            samples = nested_tensor_from_tensor_list([image_transformed.to(args.device)])
            outputs = model(samples)

        # --- Post-processing --- #
        orig_target_sizes = torch.tensor([image.size[::-1]], device=args.device)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        predictions = results[0]

        # --- Visualization --- #
        visualized_image = visualize_predictions(image, predictions, category_mapping)

        # --- Display and Save --- #
        plt.figure(figsize=(16, 10))
        plt.imshow(visualized_image)
        plt.axis('off')
        plt.show()
        
        output_path = os.path.join(output_dir, f"dino_{random_image_file}")
        visualized_image.save(output_path)
        print(f"Saved visualized image to: {output_path}")

        another = input("Show another image? (y/n): ")
        if another.lower() != 'y':
            break


if __name__ == '__main__':
    main()
