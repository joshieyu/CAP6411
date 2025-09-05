
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
import os
import random
import matplotlib.pyplot as plt
from coco_categories import COCO_CATEGORIES

def get_coco_categories():
    category_mapping = {cat['id']: cat['name'] for cat in COCO_CATEGORIES}
    return category_mapping

def visualize_predictions(image, predictions, category_mapping, threshold=0.5):
    draw = ImageDraw.Draw(image)
    
    for i in range(len(predictions[0]['labels'])):
        score = predictions[0]['scores'][i].item()
        if score > threshold:
            box = predictions[0]['boxes'][i].cpu().numpy()
            label_id = predictions[0]['labels'][i].item()
            label_name = category_mapping.get(label_id, "N/A")
            
            print(
                f"Detected {label_name} with confidence "
                f"{round(score, 3)} at location {[round(coord, 2) for coord in box]}"
            )
            
            # Draw box
            draw.rectangle(box, outline="red", width=3)
            
            # Draw label
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            
            text = f"{label_name}: {round(score, 2)}"
            text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
            draw.rectangle(text_bbox, fill="red")
            draw.text((box[0], box[1]), text, fill="white", font=font)
            
    return image

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(script_dir, 'cocoDataset', 'val2017')
    output_dir = os.path.join(script_dir, 'inference_output')
    os.makedirs(output_dir, exist_ok=True)

    # Load Model
    model = fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    # Get category mapping
    category_mapping = get_coco_categories()

    while True:
        # Get a random image
        image_files = os.listdir(img_dir)
        random_image_file = random.choice(image_files)
        img_path = os.path.join(img_dir, random_image_file)
        
        print(f"\nProcessing image: {img_path}")
        image = Image.open(img_path).convert('RGB')

        # Preprocess
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        img_tensor = transform(image).to(device)

        # Inference
        with torch.no_grad():
            predictions = model([img_tensor])

        # Visualize
        visualized_image = visualize_predictions(image, predictions, category_mapping)
        
        # Display the image
        plt.figure(figsize=(16, 10))
        plt.imshow(visualized_image)
        plt.axis('off')
        plt.show()
        
        # Save the image
        output_path = os.path.join(output_dir, f"faster_rcnn_{random_image_file}")
        visualized_image.save(output_path)
        print(f"Saved visualized image to: {output_path}")

        another = input("Show another image? (y/n): ")
        if another.lower() != 'y':
            break

if __name__ == '__main__':
    main()
