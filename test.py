# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 16:45:37 2021

@author: Sunlu Zeng
"""

import tensorflow as tf
import torch
import numpy as np
from data_loader import *
from Config import Configuration
from network import model_Seq, mymodel
from tensorflow.keras.models import load_model, save_model
from tensorflow.data import Dataset
import soundfile as sf
import sounddevice as sd
import librosa
from sklearn.metrics import mean_squared_error
from librosa.display import specshow
import matplotlib.pyplot as plt




MSE = tf.keras.losses.MeanSquaredError()
hypara = Configuration()
start_frame = hypara.start_frame
end_frame = hypara.end_frame
hop = hypara.hop_size
sr = hypara.sample_rate
num_of_test = 10
print('start loading dataset ...')
test_data = create_dataset('dataset/audio/test','dataset/video/test')
print('dataset loading completed!')
model_path = 'models/Seqmodel_mse_0.01331643108278513.h5'
print(f'loading model from {model_path}')
modelA = load_model('models/Seqmodel_mse_0.01331643108278513.h5')
modelB = load_model('models/Seqmodel_mse_0.02105019800364971.h5')
modelC = load_model('models/Seqmodel_mse_0.024626480415463448.h5')


def plot_mel(spec):
    fig, ax = plt.subplots(figsize=(10, 5), sharey=True)
    img = specshow(spec,y_axis='mel',x_axis='time',ax=ax)
    ax.set(title='Mel-Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.2f dB")
    plt.show()  


def plot_mels(spec_est,spec):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    img0 = specshow(spec_est,y_axis='mel',x_axis='time',ax=ax[0])
    img1 = specshow(spec,y_axis='mel',x_axis='time',ax=ax[1])
    ax[0].set(title='inpainted mel-spectrogram')
    ax[1].set(title='true mel-spectrogram')
    fig.colorbar(img1, ax=ax[1], format="%+2.2f dB")
    plt.show()
    


def plot_wave(wv):
    fig, ax = plt.subplots(figsize=(10, 5), sharey=True)
    ax.set(title='original speech')
    ax.plot(wv)
    plt.show()    

def plot_waves(wv,abwv):
    fig, ax = plt.subplots(2,1,figsize=(16, 9), sharey=True)
    ax[0].set(title='original speech')
    ax[0].plot(wv)
    ax[1].set(title='converted speech')
    ax[1].plot(abwv)
    plt.show()    

def wv_array(path='dataset/audio/test/s30'):
  '''
  Load audio from individual set (training, validation or test)
  Keywords arguments:
  return -- numpy array, dtype=np.float32, shape=(num_of_audio, audio_length)
  '''

  audio_arr = []
  file_list = glob(f'{path}/*.wav')
  for i,file in enumerate(file_list):
    wv, sr = librosa.load(file, 16000)
    wv = np.array(wv, dtype=np.float32)
    audio_arr.append(wv)
  return np.array(audio_arr, dtype=np.float32)    

    
print('start audio inpainting ...')
test_spec_estA = modelA.predict(test_data).transpose((0,2,1))[:num_of_test]
test_spec_estB = modelB.predict(test_data).transpose((0,2,1))[:num_of_test]
test_spec_estC = modelC.predict(test_data).transpose((0,2,1))[:num_of_test]

test_spec_true = load_mel_array('dataset/audio/test/')[:num_of_test]

plot_mel(test_spec_true[5])
plot_mel(test_spec_estA[5])
plot_mel(test_spec_estB[5])
plot_mel(test_spec_estC[5])


print('audio inpainting completed!')

#mse_loss = model.evaluate(test_data)
test_spec_true = load_mel_array('dataset/audio/test/')[:num_of_test]

plot_mels(test_spec_est[5],test_spec_true[5])
test_spec_est = librosa.db_to_power(denormalize_melspec(test_spec_est) + hypara.ref_level_db)
test_spec_true = librosa.db_to_power(denormalize_melspec(test_spec_true) + hypara.ref_level_db)
wv_true = wv_array()[:num_of_test]
for i in range(10):
    wv_crpt = np.copy(wv_true[i])
    wv_crpt[start_frame*hop:end_frame*hop] = -np.ones(((end_frame-start_frame)*hop))
    wv_est = spec_to_audio(np.expand_dims(test_spec_est[i],0), melspecfunc, maxiter=500, evaiter=10, tol=1e-8)
    wv_est[:start_frame*hop] = wv_true[i,:start_frame*hop]
    wv_est[end_frame*hop:] = wv_true[i,end_frame*hop:]
    
    print('playing corrupted speech')
    sd.play(wv_crpt,sr)
    input()

    print('playing inpainted speech')
    sd.play(wv_est,sr)
    input()
    
    print('playing original speech')
    sd.play(wv_true[i],sr)
    input()
    
    
    
#mse_loss = model.evaluate(test_data)


#test_spec_est = inmelfunc(torch.tensor(test_spec_est))



#print(f'mse is {mse_loss}')
'''
for idx in range(2):
    print(idx)
    
    wv_est = librosa.util.normalize(wv_est)
    sf.write(f'results/{idx}.wav',wv_est,hypara.sample_rate)
'''