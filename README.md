# Facial Expression Recognition

## Overview

This project implements a **Facial Expression Recognition** system using a **Convolutional Neural Network (CNN)**. It classifies human facial expressions into seven categories: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**.  

The project was developed as part of an **Artificial Intelligence Internship at Coding Samurai**.

---

## Dataset

The model is trained on the **<a href="https://www.kaggle.com/datasets/msambare/fer2013" target="_blank">FER2013 Dataset on Kaggle</a>**:

- ~35,000 grayscale images of size 48×48 pixels  
- Each image is labeled with one of seven emotions  

---

## GitHub Repository Structure
```
Facial-Expression-Recognition/
│
├── other/Face_expression_recognition.ipynb        # Original Colab notebook
├── Face_expression_recognition_notebook.ipynb     # Cleaned notebook for GitHub
├── emotion_detection.py                            # Real-time OpenCV detection script
├── other/train/emotiondetector.keras              # Trained CNN model
├── requirements.txt                               # Python dependencies
└── README.md                                      # Project documentation
```

---

## Project Workflow

### 1. Training the Model

- Training was performed in **Google Colab** to leverage GPU acceleration.  
- Dataset was downloaded from Kaggle, converted into a **DataFrame**, and features were extracted from images.  
- The **CNN model** was built using TensorFlow/Keras with layers including convolution, batch normalization, pooling, dropout, and softmax classifier.  

**Training process:**  
- First round: 150 epochs, batch size 31 → ~65% accuracy  
- Applied **callbacks** (EarlyStopping, ReduceLROnPlateau) and **data augmentation** (rotation, shifting, zoom, horizontal flip)  
- Second round: 150 epochs, batch size 64  
- Third round: 150 epochs, batch size 64 (continuous fitting)  
- Final accuracy: **66.34%**  

> Some predictions were initially incorrect (e.g., anger predicted as fear, neutral as angry), which improved after augmentation and callbacks.

The trained model was exported as a `.keras` file for testing.  

**Full training notebook:**  
<a href="https://github.com/itsmenisha/Facial-Expression-Recognition/blob/main/Face_expression_recognition_notebook.ipynb" target="_blank">Face_expression_recognition_notebook.ipynb</a>


---

### 2. Real-Time Testing

- The trained model was tested using **OpenCV** in **Visual Studio Code**.  
- Faces were detected with Haar cascades, converted to grayscale, resized to 48×48 pixels, and fed into the CNN for emotion recognition.  
- Real-time webcam predictions are displayed with bounding boxes and emotion labels.  

**<a href="https://github.com/itsmenisha/Facial-Expression-Recognition/blob/main/Face_expression_recognition_notebook.ipynb" target="_blank">Real-time detection code</a>**  

---

### 3. Cleaning the Notebook

- To share the Colab notebook on GitHub, widgets were removed using a Python script that cleans notebook metadata.  
- The cleaned notebook is available in the repository.
<a href="https://github.com/itsmenisha/Facial-Expression-Recognition/blob/main/real_time_emotion_detection.py" target="_blank">real_time_emotion_detection code</a>


---

## Installation & Requirements

Ensure **Python 3.8+** is installed. Install dependencies:

```bash
pip install -r requirements.txt
```
---
## Cloning the Repository

```bash
git clone https://github.com/itsmenisha/Facial-Expression-Recognition.git
cd Facial-Expression-Recognition
```
---
## Usage

- Open the Colab notebook and run all cells to train the CNN with the FER2013 dataset with GPU.
- Run the detection script with a connected webcam:
bash'''
python real_time_emotion_detection.py
'''

---

## Tools & Technologies

- Google Colab – GPU training

- Python – Core programming language

- TensorFlow/Keras – CNN model building and training

- OpenCV – Real-time face detection and emotion recognition

- Kaggle FER2013 Dataset – Training dataset

---
