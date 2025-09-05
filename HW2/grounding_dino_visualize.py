
import os
import sys
import argparse
import random
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import requests

# Add GroundingDINO project to python path
script_dir = os.path.dirname(os.path.abspath(__file__))
grounding_dino_dir = os.path.join(script_dir, 'GroundingDINO')
sys.path.append(grounding_dino_dir)

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
import groundingdino.datasets.transforms as T
from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.util.utils import get_phrases_from_posmap

def download_file(url, local_filename):
    if os.path.exists(local_filename):
        print(f"{local_filename} already exists. Skipping download.")
        return
    print(f"Downloading {local_filename}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)

    for box, label in zip(boxes, labels):
        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        x0, y0, x1, y1 = [int(i) for i in box]

        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        text = str(label)
        text_bbox = draw.textbbox((x0, y0), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x0, y0), text, fill="white", font=font)

    return image_pil

def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
        
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]
    
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]
    
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase + f"({logit.max().item():.2f})")
        
    return boxes_filt.cpu(), pred_phrases

def main():
    # --- Config ---
    config_file = os.path.join(grounding_dino_dir, 'groundingdino/config/GroundingDINO_SwinT_OGC.py')
    weights_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    weights_dir = os.path.join(grounding_dino_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    checkpoint_path = os.path.join(weights_dir, 'groundingdino_swint_ogc.pth')
    img_dir = os.path.join(script_dir, 'cocoDataset', 'val2017')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Download weights ---
    download_file(weights_url, checkpoint_path)

    # --- Load Model ---
    model = load_model(config_file, checkpoint_path, device)

    while True:
        # --- Image Selection and Prompting ---
        image_files = os.listdir(img_dir)
        random_image_file = random.choice(image_files)
        img_path = os.path.join(img_dir, random_image_file)
        print(f"\nSelected image: {img_path}")
        
        image_pil, image = load_image(img_path)

        # Display the image first
        plt.figure(figsize=(16, 10))
        plt.imshow(image_pil)
        plt.axis('off')
        plt.title("Close this window to enter a text prompt")
        plt.show()

        text_prompt = input("Enter the text prompt (e.g., 'a cat and a dog'): ")
        if not text_prompt:
            print("No prompt entered, exiting.")
            break

        # --- Inference ---
        boxes, labels = get_grounding_output(model, image, text_prompt, 0.3, 0.25, device)

        # --- Visualization ---
        pred_dict = {
            "boxes": boxes,
            "size": image_pil.size[::-1],  # H,W
            "labels": labels,
        }
        visualized_image = plot_boxes_to_image(image_pil, pred_dict)

        # --- Display and Save ---
        plt.figure(figsize=(16, 10))
        plt.imshow(visualized_image)
        plt.axis('off')
        plt.title(f"Result for: '{text_prompt}'")
        plt.show()
        
        output_dir = os.path.join(script_dir, 'inference_output')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"grounding_dino_{random_image_file}")
        visualized_image.save(output_path)
        print(f"Saved visualized image to: {output_path}")

        another = input("Show another image? (y/n): ")
        if another.lower() != 'y':
            break


if __name__ == '__main__':
    main()
