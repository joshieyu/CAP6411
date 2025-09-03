import os
import subprocess
import requests

def download_file(url, local_filename):
    print(f"Downloading {local_filename}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

def main():
    print("--- Running Grounding DINO Evaluation ---")
    repo_dir = 'GroundingDINO'
    
    # The repo expects to be run from its own directory
    os.chdir(repo_dir)

    # Create weights directory and download weights
    os.makedirs('weights', exist_ok=True)
    weights_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    weights_path = os.path.join('weights', 'groundingdino_swint_ogc.pth')
    if not os.path.exists(weights_path):
        download_file(weights_url, weights_path)

    # The official evaluation command
    config_file = 'groundingdino/config/GroundingDINO_SwinT_OGC.py'
    coco_path = os.path.abspath(os.path.join('..', 'cocoDataset'))

    eval_command = [
        'python', 'tools/eval.py',
        '--config-file', config_file,
        '--resume', weights_path,
        '--eval-only',
        '--gpus', '1',
        '--num-machines', '1',
        '--machine-rank', '0',
        f'--options', f'data.test.img_folder={os.path.join(coco_path, "val2017")}',
        f'data.test.ann_file={os.path.join(coco_path, "annotations", "instances_val2017.json")}'
    ]

    print(f"Running command: {' '.join(eval_command)}")
    try:
        subprocess.run(eval_command, check=True)
        print("--- Grounding DINO Evaluation Complete ---")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running Grounding DINO evaluation: {e}")
        print("Please ensure all dependencies from the GroundingDINO repository are installed.")

if __name__ == "__main__":
    main()
