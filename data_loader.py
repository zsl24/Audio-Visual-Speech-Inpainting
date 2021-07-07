import tensorflow as tf
import numpy as np
from glob import glob
import os
import librosa
from Config import Configuration
from time import time
import torch
from torchaudio.transforms import MelScale, Spectrogram


hypara = Configuration()
root_path = hypara.root_path

specobj = Spectrogram(n_fft=hypara.stft_size, win_length=hypara.win_size, hop_length=hypara.hop_size, pad=0, power=2, normalized=True)
specfunc = specobj.forward

def audio_array(path):
  '''
  Load audio from individual set (training, validation or test)
  '''
  folder_list = os.listdir(path)
  file_list = []
  for folder in folder_list:
    file_list.extend(glob(f'{path}/{folder}/*.wav'))

  audio_arr = []
  for i in range(10):
    wv, sr = librosa.load(file_list[i], 25000)
    wv = np.array(wv, dtype=np.float32)
    audio_arr.append(wv)
  return np.array(audio_arr)

def audio_to_spec(wv_array):
  ''' 
  Convert audio array into spectrogram array
  input: numpy array, dtype=object, shape=(number of audios, )
  return numpy array, dtype=object, shape=(number of spectrograms, )
  '''

  num_of_samples = wv_array.shape[0]
  spec_arr = np.empty(num_of_samples, dtype=object)
  for i in range(num_of_samples):
    wv = wv_array[i]
    spec = np.array(specfunc(torch.tensor(wv_array[i])).detach().cpu(), dtype=np.float32)
    spec_arr[i] = np.expand_dims(spec, -1)
  return spec_arr

def corruption_spec(spec_arr, start_frame=80, end_frame=90):
  '''
  Add corrpution to spectrogram array with many spectrograms given corrupted position in time frame
  '''
  num_of_samples = spec_arr.shape[0]
  freq_bins = spec_arr[0].shape[0]
  crpt_length = end_frame-start_frame
  z = np.zeros((freq_bins, crpt_length, 1))
  for i in range(num_of_samples):
      spec_arr[i][:,start_frame:end_frame] *= z
  return spec_arr




    
  

if __name__ == '__main__':
  pass
  adata = audio_array(root_path+'dataset/audio/train/')
  spec_data = audio_to_spec(adata)
  c_spec_data = corruption_spec(spec_data)
  train_A_Y = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(spec_data)).batch(hypara.batch_size, drop_remainder=True)

