# Adversarial-ML (CV) Overview


This project demonstrates an existing method highlighted in the paper below, 

https://openaccess.thecvf.com/content_CVPRW_2019/papers/CV-COPS/Thys_Fooling_Automated_Surveillance_Cameras_Adversarial_Patches_to_Attack_Person_Detection_CVPRW_2019_paper.pdf

which shows how small changes to a particular image (either physically or digitally) can drastically lower a model's confidence level on what it is intended to predict.

# How it Works

Freezes the model's weights (by setting required_grad=False) while using gradient descent to decrease loss as the person's confidence score decreases.

# Usage: Training

1.) Clone the repo
2.) Create a python virtual environment (python -m venv name_of_venv)
3.) Install dependencies (pip install -r requirements.txt)
4.) *Download preferred dataset (Roboflow, Kaggle, etc.) and save it to adv-patch/
5.) Run train.py (key CLI flags are --epochs, --tv_weight, --out, --seed) 
6.) each epoch saves a .pt and .png version of the patch

*Dataset.py designed for extracting from Roboflow for YOLO specifically, modifications will be necessary for non-YOLO or non-Roboflow sources


# Usage: Validation
1.) *Run eval.py (key CLI flags of --n-trials, --seed, --scale, --patch) on the .pt you generated
2.) use make_control_patch.py to generate a gray control
3.) Run eval.py on the gray control's .pt file
4.) Compare the model's dropoff between the two outputs.



*The seed MUST be the same for both training and validation or significant change will be attributed to random chance.