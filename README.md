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

There are 33 speakers in this dataset, S1~S20, S20~S34 (Video data for S21 is not available). For each speaker, 1000 pairs of audio video about different uttering sentences are included. (Actually, some videos in speaker number 15, S15, are blurry and not able to detector facial landmarks, the list of file names are included in file errorlog.txt in this repository)

We followed data spliting strategy that assigns 25 speakers (13 male, 12 female, S1-S20, S22-25, S28) for training, 4 speakers (2 male, 2 female, S26-S27, S29, S31) for validation and the rest 4 (2 male, 2 female, S30, S32-S34) for testing.  

## Feature maps for both audio and video
Because feature extractions for both audio and video are time consuming, we cannot do feature extraction every time we train. Therefore, before training, we extract the feature map and save them as numpy array in .npy files.  

The path 
>


## How to extract facial landmarks using dlib
- place facial landmark detector model file, shape_predictor_68_face_landmarks.dat, to root path of this project
- create faical landmark detector object in your Python code: dlib.shape_detector(path='shape_predictor_68_face_landmarks.dat')


