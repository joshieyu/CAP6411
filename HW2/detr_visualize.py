
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import os
import random
import matplotlib.pyplot as plt
from coco_categories import COCO_CATEGORIES

def get_coco_categories():
    # The model was trained on COCO 2017, which has 80 classes.
    # The transformers library maps these to 91 classes.
    # We can create a mapping from the model's output category ID to the category name.
    
    # The model's category IDs are 1-based and seem to align with the COCO dataset's IDs.
    # However, the DetrForObjectDetection model from huggingface transformers has a background class at index 91.
    # The model's output labels are 1-indexed, so we can use them directly with the COCO category IDs.
    
    category_mapping = {cat['id']: cat['name'] for cat in COCO_CATEGORIES}
    return category_mapping

def visualize_predictions(image, predictions, category_mapping, threshold=0.9):
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
            
            # Draw box
            draw.rectangle(box, outline="red", width=3)
            
            # Draw label
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
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(script_dir, 'cocoDataset', 'val2017')
    output_dir = os.path.join(script_dir, 'inference_output')
    os.makedirs(output_dir, exist_ok=True)

    # Load Model
    processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
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
        inputs = processor(images=image, return_tensors='pt').to(device)

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Format results
        target_sizes = torch.tensor([image.size[::-1]])
        predictions = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        # Visualize
        visualized_image = visualize_predictions(image, predictions, category_mapping)
        
        # Display the image
        plt.figure(figsize=(16, 10))
        plt.imshow(visualized_image)
        plt.axis('off')
        plt.show()
        
        # Save the image
        output_path = os.path.join(output_dir, f"detr_{random_image_file}")
        visualized_image.save(output_path)
        print(f"Saved visualized image to: {output_path}")

        another = input("Show another image? (y/n): ")
        if another.lower() != 'y':
            break



if __name__ == '__main__':
    main()
