# DLP
## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Results](#results)
- [Dependencies](#dependencies)

## Overview
This is an end-to-end deep learning project to recognize and classify traffic signs. It uses a CNN model and a ViT trained on a labeled dataset consisting of images and metadata and compares the two.
## Dataset
- Located in the `data/` folder
- Performed data augmentation on the train dataset.
reference: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
<pre> ## Project Structure ``` DLP/ │ ├── data/ # Contains traffic sign dataset files and folders │ ├── 0/ │ ├── 1/ │ ├── ..... │ ├── alldirection.py # Script for detecting all traffic sign directions ├── requiremnts.txt # dependencies ├── main.py # Main training or evaluation script ├── vit.py # Vision Transformer model code ├── README.md # Project documentation ``` </pre>
## How to Run
1. Clone the repository
git clone https://github.com/your-username/traffic-sign-detection.git
cd traffic-sign-detection

2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate

3. Install dependencies
Make sure you have Python 3.8+ installed.
pip install -r requirements.txt

4. Prepare the dataset
Place your dataset inside the data/ directory

5. run alldirection.py
   run python main.py -CNN
   run python vit.py  -ViT
## Results
The ViT model clearly outperformed the CNN counterpart in both accuracy and generalization. Its ability to learn global features via self-attention likely contributed to this boost in performance, making it a highly suitable candidate for advanced driver assistance systems and real-time traffic sign recognition in autonomous vehicles.

## Dependencies
pip install -r requirements.txt



   




