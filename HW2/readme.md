VIDEO LINK:

https://youtu.be/jwuDmlolCv8


This project contains scripts to evaluate and visualize different object detection models.

## Setup

You can set up the environment using either Conda or pip, but I have not tested Conda personally.
I had trouble setting up with Conda last time but pip worked.

### Pip (Recommended)

1.  Create a virtual environment:
    ```
    python -m venv CVENV
    source CVENV/bin/activate  # On Windows, use `CVENV\Scripts\activate`
    ```
2.  Install the dependencies from `requirements.txt`:
    ```
    pip install -r requirements.txt
    ```
3.  Install the `groundingdino` package manually. From the `GroundingDINO` directory, run:
    ```
    pip install -e . --no-build-isolation
    ```
    *Note*: The `setup.py` in `GroundingDINO` has been modified to prevent it from automatically trying to install `torch`. If you re-clone the repository, you may need to comment out the `install_torch()` call in `setup.py`.

### Conda

1.  Create the conda environment from the `environment.yml` file:
    ```
    conda env create -f environment.yml
    ```
2.  Activate the environment:
    ```
    conda activate CVENV
    ```
3.  Optional: Install the `groundingdino` package manually. From the `GroundingDINO` directory, run:
    ```
    pip install -e . --no-build-isolation
    ```
    *Note*: The `setup.py` in `GroundingDINO` has been modified to prevent it from automatically trying to install `torch`. If you re-clone the repository, you may need to comment out the `install_torch()` call in `setup.py`.


## Model Visualization (Inference)

These scripts allow you to visualize the output of the models on random images from the COCO validation set. The scripts will run in a loop, allowing you to view multiple images without reloading the models.

-   **Faster R-CNN**:
    ```
    python faster_rcnn_visualize.py
    ```
-   **DETR**:
    ```
    python detr_visualize.py
    ```
-   **DINO**:
    ```
    Open DINO/inference_and_visualization.ipynb
    Run all cells
    ```
-   **GroundingDINO**:
    ```
    python grounding_dino_visualize.py
    ```

## Model Evaluation

These scripts run the models on the COCO dataset and save the results.

-   **Run Faster R-CNN evaluation**:
    ```
    python faster_rcnn_inference.py
    ```
-   **Run DETR evaluation**:
    ```
    python detr_inference.py
    ```
-   **Run DINO evaluation**:
    ```
    python run_dino_eval.py
    ```
-   **Run GroundingDINO evaluation**:
    ```
    python run_grounding_dino_eval.py
    ```

