# Import the required libraries.
import os
import cv2
# import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
# import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from moviepy.editor import *


from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import streamlit as st
# import subprocess
import base64
import pickle

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 20

# Specify the directory containing the UCF50 dataset. 
DATASET_DIR = "dataset/UCF101"

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = pickle.load(open('filenames.pkl','rb'))
# ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]


# # Make the Output directory if it does not exist
# test_videos_directory = 'test_videos'
# # os.makedirs(test_videos_directory, exist_ok = True)

# Load the saved model
loaded_model = load_model('savedModel/LRCN_model___Date_Time_2024_01_29__16_56_04___Loss_0.5569551587104797___Accuracy_0.8395990133285522.h5')


def save_uploaded_file(uploaded_file):
    # Create a folder to store uploaded videos if it doesn't exist
    if not os.path.exists("test_videos"):
        os.makedirs("test_videos")

    # Save the uploaded video to the "uploaded_videos" folder
    with open(os.path.join("test_videos", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    # st.success(f"Video '{uploaded_file.name}' saved successfully!")





# Create a Function To Perform Action Recognition on Videos
def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
    '''
    This function will perform action recognition on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    output_file_path: The path where the ouput video with the predicted action being performed overlayed will be stored.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoWriter Object to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():

        # Read the frame.
        ok, frame = video_reader.read() 
        
        # Check if frame is not read properly then break the loop.
        if not ok:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:

            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = loaded_model.predict(np.expand_dims(frames_queue, axis = 0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]
            
            
        # Write predicted class name on top of the frame.
        cv2.putText(frame, predicted_class_name, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        # Write The frame into the disk using the VideoWriter Object.
        video_writer.write(frame)
        
    # Release the VideoCapture and VideoWriter objects.
    video_reader.release()
    video_writer.release()

# # Make the Output directory if it does not exist
# test_videos_directory = 'test_videos'
# os.makedirs(test_videos_directory, exist_ok = True)


# # Display the output video.
# VideoFileClip(output_video_file_path, audio=False, target_resolution=(300,None)).ipython_display()


def main():
    st.title("Action Recognition System")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
    
    if uploaded_file:
        st.video(uploaded_file)
        # st.write("Filename: ", uploaded_file.name)
        save_uploaded_file(uploaded_file)
                
        video_title = uploaded_file.name.split(".")[0]
        # uploaded_file.name
        test_videos_directory = 'test_videos'
        
        output_video_directory = 'output_video'
        # Construct the output video path.
        output_video_file_path = f'{output_video_directory}/{video_title}-Output-SeqLen{SEQUENCE_LENGTH}.mp4'

        input_video_file_path = f'{test_videos_directory}/{video_title}.mp4'
        button_clicked = st.button("Predict the action")
        # Check if the button is clicked
        
        if button_clicked:
        # Perform Action Recognition on the Test Video.
            predict_on_video(input_video_file_path, output_video_file_path, SEQUENCE_LENGTH)
        
            print(output_video_file_path)
             
             # Specify the path to your video file
            video_path = output_video_file_path

            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                st.error("Error: Unable to open the video file.")
                return

            # Get the video properties (width, height, frames per second, etc.)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

             # Display the video in the Streamlit app
            stframe = st.image([])

            while True:
                # Read a frame from the video
                ret, frame = cap.read()

                if not ret:
                    # Break the loop if no more frames are available
                    break

                # Convert the OpenCV frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display the frame in the Streamlit app
                stframe.image(rgb_frame, channels="RGB")


            # Create a download link for the video
            download_text = "Download Video"
            st.markdown(get_download_link(video_path, download_text), unsafe_allow_html=True)

           

            # Release the VideoCapture object
            cap.release()

def get_download_link(file_path, download_text):
    """
    Generate a download link for a file.
    """
    with open(file_path, 'rb') as file:
        file_content = file.read()
        base64_encoded = base64.b64encode(file_content).decode()
        href = f'<a href="data:video/mp4;base64,{base64_encoded}" download="{download_text}.mp4">{download_text}</a>'
    return href
           

if __name__ == "__main__":
    main()













