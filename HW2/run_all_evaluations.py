import os
import subprocess
import git

def clone_repo(repo_url, repo_dir):
    if not os.path.exists(repo_dir):
        print(f"Cloning {repo_url} into {repo_dir}...")
        git.Repo.clone_from(repo_url, repo_dir)
        print("Clone complete.")
    else:
        print(f"Directory {repo_dir} already exists. Skipping clone.")

def run_script(script_name):
    print(f"\n{'='*20} Running {script_name} {'='*20}")
    try:
        subprocess.run(['python', script_name], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running {script_name}: {e}")
    print(f"{'='*20} Finished {script_name} {'='*20}\n")

def main():
    # --- Setup: Clone Repositories ---
    print("--- Setting up external repositories ---")
    # Note: The official DINO object detection repo is not as straightforward as GroundingDINO.
    # We are cloning the base DINO repo as a placeholder.
    clone_repo("https://github.com/IDEA-Research/DINO.git", "DINO")
    clone_repo("https://github.com/IDEA-Research/GroundingDINO.git", "GroundingDINO")
    print("--- Repository setup complete ---")

    # --- Run Evaluations ---
    # 1. Faster R-CNN (Hugging Face)
    # run_script('faster_rcnn_inference.py')

    # 2. DETR (Hugging Face)
    # run_script('detr_inference.py')

    # 3. DINO (from repository - placeholder)
    # The run_dino_eval.py script needs to be configured with correct weights and commands
    # based on the specific DINO object detection model you choose to use.
    # run_script('run_dino_eval.py')

    # 4. Grounding DINO (from repository)
    # run_script('run_grounding_dino_eval.py')

    print("All evaluations complete.")

if __name__ == "__main__":
    # Change to HW2 directory to run everything
    if os.path.basename(os.getcwd()) != 'HW2':
        os.chdir('HW2')
    main()
