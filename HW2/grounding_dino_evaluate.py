import os
import subprocess
import requests

def download_file(url, local_filename):
    # Check if the file already exists and is not empty
    if os.path.exists(local_filename) and os.path.getsize(local_filename) > 0:
        print(f"{os.path.basename(local_filename)} already exists. Skipping download.")
        return
    
    print(f"Downloading {os.path.basename(url)} to {local_filename}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

def main():
    print("--- Running Grounding DINO Evaluation ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.join(script_dir, 'GroundingDINO')

    # Construct the path to the python executable in the virtual environment
    python_executable = os.path.abspath(os.path.join(script_dir, '..', 'CVENV', 'Scripts', 'python.exe'))
    
    original_cwd = os.getcwd()
    try:
        # The repo expects to be run from its own directory
        os.chdir(repo_dir)

        # --- Download weights ---
        weights_dir = 'weights'
        os.makedirs(weights_dir, exist_ok=True)
        weights_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        weights_path = os.path.join(weights_dir, 'groundingdino_swint_ogc.pth')
        download_file(weights_url, weights_path)

        # --- Prepare command ---
        # Paths are relative to the 'GroundingDINO' directory
        eval_script = os.path.join('demo', 'test_ap_on_coco.py')
        config_file = os.path.join('groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py')
        
        coco_path = os.path.join('..', 'cocoDataset')
        anno_path = os.path.join(coco_path, "annotations", "instances_val2017.json")
        image_dir = os.path.join(coco_path, "val2017")

        eval_command = [
            python_executable, eval_script,
            '-c', config_file,
            '-p', weights_path,
            '--anno_path', anno_path,
            '--image_dir', image_dir
        ]

        print(f"Running command: {' '.join(eval_command)}")
        
        # --- Run evaluation ---
        subprocess.run(eval_command, check=True)
        
        print("--- Grounding DINO Evaluation Complete ---")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running Grounding DINO evaluation: {e}")
        print("Please ensure all dependencies from the GroundingDINO repository are installed.")
    finally:
        # Return to the original directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()