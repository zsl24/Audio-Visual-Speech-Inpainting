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
- Download dlib package from https://github.com/zsl24/face_recog_dlib_file
- unzip the package
- pip install wheel:
  - for python 3.8, please enter pip command: pip install dlib-19.19.0-cp38-cp38-win_amd64.whl
  - for python 3.7, change 38 to 37
- Install Cmake and add its path to system path: C:/ProgramFile/Cmake/Bin
- pip install cmake

## Audio-Visual Dataset: GRID Corpus
Please download from: [GRID Corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/)

## How to extract facial landmarks using dlib
- place facial landmark detector model file, shape_predictor_68_face_landmarks.dat, to root file of this project
- create faical landmark detector object in your Python code: dlib.shape_detector(path='shape_predictor_68_face_landmarks.dat')


