# Audio-Visual-Speech-Inpainter
Demo for speech inpainting  
Reference paper: [Audio-Visual Speech Inpainting with Deep Learning](https://arxiv.org/abs/2010.04556)  
Note: The code is not completed for now.  

## Package Requirement
python 3.7  
tensorflow 2.4.0  
numpy 1.19.2 (this is the version compatible with both given version of tensorflow and pytorch)  
librosa 0.8.1  
sounddevice 0.4.2 
pytorch 1.7.1 (torchaudio 0.7.2)  
cuda 11.0  
cudnn 8.04  
opencv  
imutils 0.5.4  


## Configuration of dlib
we use dlib to extract facial landmarks from video of speakers  
- Download dlib package from https://github.com/zsl24/face_recog_dlib_file
- unzip the package
- pip install wheel:
  - for python 3.8, please enter pip command: pip install dlib-19.19.0-cp38-cp38-win_amd64.whl
  - for python 3.7, change 38 to 37
- Install Cmake and add its path to system path: C:/ProgramFile/Cmake/Bin
- pip install cmake

## Audio-Visual Dataset: GRID Corpus
Please download from: [GRID Corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/)  
We collect audio visual human speech dataset GRID Corpus as the one for training, validation and test for this supervised learning task. This dataset contains both audio and video of people uttering short sentences. The audio is recorded in very silent recording studio which make inferencing noise impossible. The face of speaking person is centered in the video frame, and we can clear see very detailed movement of the face in the video.  

In order to conduct supervised learning, dataset splitting is required. We divide this dataset into 3 parts, training set, validation set and test set. We assign 25 speakers (13 male, 12 female) for training, 4 speakers (2 male, 2 female) for validation and the rest 4 (2 male, 2 female) for testing. Basically, all parts of this dataset show no gender difference, which will train models that are unbiased to the gender of the speaker.  

## Feature maps for both audio and video
Because feature extractions for both audio and video are time consuming, we cannot do feature extraction every time we train. Therefore, before training, we extract the feature map and save them as numpy array in .npy files.


## How to extract facial landmarks using dlib
- place facial landmark detector model file, shape_predictor_68_face_landmarks.dat, to root path of this project
- create faical landmark detector object in your Python code: dlib.shape_detector(path='shape_predictor_68_face_landmarks.dat')


