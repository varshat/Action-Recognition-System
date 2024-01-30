# Action Recognition System using LRCN Approach
## Overview
This project implements a Park Action Recognition System utilizing the Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) Approach. The system is designed to analyze video data captured in parks and recognize various activities, including walking, jogging, exercising, and playing.

This project implements a Garden Activity Recognition System using approach known as the Long-term Recurrent Convolutional Network (LRCN), which combines CNN and LSTM layers in a single model. 

## Features
LRCN Model:  the Long-term Recurrent Convolutional Network (LRCN), which combines CNN and LSTM layers in a single model. The Convolutional layers are used for spatial feature extraction from the frames, and the extracted spatial features are fed to LSTM layer(s) at each time-steps for temporal sequence modeling. This way the network learns spatiotemporal features directly in an end-to-end training, resulting in a robust model.

Dataset: https://www.crcv.ucf.edu/data/UCF101.php which includes a sample dataset of garden activities for training and evaluation.

User Interface: A simple user interface for real-time monitoring and visualization of recognized activities.

## Setup
Clone the Repository:
git clone https://github.com/your-username/garden-activity-recognition.git
cd garden-activity-recognition

Install Dependencies:
pip install -r requirements.txt

Run the Application:
python main.py

Access the UI:
Open your web browser and navigate to http://localhost:5000 to access the real-time monitoring interface.

Usage
Training:
Prepare your dataset and organize it following the provided format.
Run the training script to train the LRCN model.

Output:


https://github.com/varshat/Action-Recognition-System/assets/7055503/673c433e-b874-4709-826a-387ccbbf6a07



License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Mention any additional libraries, tools, or resources used in the project.
