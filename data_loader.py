import tensorflow as tf
import numpy as np
from glob import glob
import os
import librosa
from config import Configuration
from time import time
import torch
from torchaudio.transforms import MelScale, Spectrogram, InverseMelScale, GriffinLim
from face_landmarks import extract_face_landmarks, show_face_landmarks, save_face_landmarks_speaker
from tqdm import tqdm
from sklearn import preprocessing
from tensorflow.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

hypara = Configuration()
#
specobj = Spectrogram(n_fft=hypara.stft_size, win_length=hypara.win_size, hop_length=hypara.hop_size, pad=0, power=2, normalized=True)
specfunc = specobj.forward
# MelScale object to transform spectrogram (shape = (n_stft//2+1,time_frames)) into Melspectrogram (shape = (n_mels,time_frames))
melobj = MelScale(n_mels=hypara.n_mels, sample_rate=hypara.sample_rate, f_min=0.)
melfunc = melobj.forward
# Inverse MelScale object to transform Melspectrogram back to spectrogram
inmelobj = InverseMelScale(n_mels=hypara.n_mels, n_stft=hypara.stft_size//2+1, sample_rate=hypara.sample_rate, f_min=0.) # the n_stft is not the correct one, it should be true n_stft//2 + 1
inmelfunc = inmelobj.forward
predictor_params = f'shape_predictor_68_face_landmarks.dat'

GLobj = GriffinLim(n_fft=hypara.stft_size, win_length=hypara.win_size, hop_length=hypara.hop_size)
GLfunc = GLobj.forward

def audio_array(path='dataset/audio/train/'):
  '''
  Load audio from individual set (training, validation or test)
  Keywords arguments:
  return -- numpy array, dtype=np.float32, shape=(num_of_audio, audio_length)
  '''
  folder_list = os.listdir(path)
  num_of_folder = min(hypara.num_of_speaker,len(folder_list))
  folder_list = folder_list[:num_of_folder]
  audio_arr = []
  for folder in tqdm(folder_list):
    file_list = glob(f'{path}/{folder}/*.wav')
    for i,file in enumerate(file_list):
      wv, sr = librosa.load(file, 16000)
      wv = np.array(wv, dtype=np.float32)
      audio_arr.append(wv)
  return np.array(audio_arr, dtype=np.float32)

def audio_to_melspec(wv_array):
  ''' Convert audio array into spectrogram array
  Keywords arguments:
  wv_array --  numpy array, dtype=object, shape=(number of audios, audio_length)
  return -- numpy array, dtype=object, shape=(number of spectrograms, freq_bins=stft_size//2+1, time_frames=audio_length//hop_size+1)
  '''
  num_of_audio, audio_length = wv_array.shape
  #freq_bins = int(hypara.stft_size//2 + 1)
  freq_bins = hypara.n_mels
  time_frames = int(audio_length//hypara.hop_size + 1)
  melspec_arr = np.empty((num_of_audio,freq_bins,time_frames), dtype=np.float32)
  for i in range(num_of_audio):
    wv = wv_array[i]
    # mel-spectrogram
    melspec = np.array(melfunc(specfunc(torch.tensor(wv))).detach().cpu(), dtype=np.float32)
    melspec = librosa.power_to_db(melspec) - hypara.ref_level_db
    melspec = normalize_melspec(melspec) # normalize to (-1,1)
    melspec_arr[i] = melspec
  return melspec_arr

def normalize_melspec(melspec):
  return np.clip((((melspec - hypara.min_level_db) / -hypara.min_level_db)*2.)-1., -1, 1)

def denormalize_melspec(melspec):
  return (((np.clip(melspec, -1, 1)+1.)/2.) * -hypara.min_level_db) + hypara.min_level_db


def corruption_spec(spec_arr, start_frame=120, end_frame=170):
  '''Add corrpution to spectrogram array with many spectrograms given corrupted position in time frame
  Keywords arguments:
  input --       numpy array, dtype=np.float32, shape=(number of audios, audio_length)
  start_frame -- int, start frame of blocked part in spectrogram
  end_frame --   int, end frame of blocked part in spectrogram
  return --      numpy array, dtype=np.float32, shape=(number of audios, audio_length)
  '''
  num_of_samples,freq_bins,time_frames = spec_arr.shape
  spec_crpt_arr = spec_arr.copy()
  crpt_length = end_frame-start_frame
  z = - np.ones((freq_bins, crpt_length))
  for i in range(num_of_samples):
      spec_crpt_arr[i][:,start_frame:end_frame] = z
  return spec_crpt_arr

def video_array(num_of_frames=251, path='dataset/video/train'):
  '''Load video from individual set (training, validation or test) and return upsampled facial landmarks array
  Keywords arguments:
  return -- numpy array, dtype=np.float32, shape=(number of audios, audio_length)
  '''
  folder_list = os.listdir(path)
  num_of_folder = len(folder_list)
  num_of_file = hypara.number_of_file_per_speaker
  video_arr = []
  print('start loading video files')
  for folder_idx,folder in enumerate(folder_list):
    file_list = glob(f'{path}/{folder}/*.mpg')
    file_list = file_list[:num_of_file]
    print(f'processing {folder}')
    for file_idx,file in enumerate(tqdm(file_list)):  
      landmarks, face_rects  = extract_face_landmarks(file, predictor_params, refresh_size=8) # landmarks.shape = (75,68,2)
      video_frames,coordinates,axises = landmarks.shape
      ldmks = np.empty((136,video_frames),dtype=np.float32)
      ldmks[:coordinates,:] = np.transpose(landmarks[:,:,0])
      ldmks[coordinates:,:] = np.transpose(landmarks[:,:,1])    # ldmk.shape = (136,75) we need to upsample to (136,251)
      ldmks = get_motion_vector(ldmks)                          # get motion vector
      ldmks_motion = resample_video_vector(ldmks,num_of_frames)
      ldmks_motion_norm = normalize_video_vector(ldmks_motion)
      video_arr.append(ldmks_motion_norm)
  return np.array(video_arr,dtype=np.float32)

def load_mel_array(path='dataset/video/train'):
    file_list = glob(f'{path}/*.npy')
    audio_arr = []
    num_of_folder = min(hypara.num_of_speaker,len(file_list))
    file_list = file_list[:num_of_folder]
    for file in tqdm(file_list):
        audio_arr.extend(np.load(file))
    return np.array(audio_arr,dtype=np.float32)

def load_video_array(path='dataset/video/train'):
    file_list = glob(f'{path}/*.npy')
    video_arr = []
    num_of_folder = min(hypara.num_of_speaker,len(file_list))
    file_list = file_list[:num_of_folder]
    for file in tqdm(file_list):
        video_arr.extend(np.load(file))
    return np.array(video_arr,dtype=np.float32)
    

def get_motion_vector(vector):
    '''get motion vector given facial landmarks vector of one video file
    Keywords arguments:
    vector -- numpy array, dtype=np.float32, shape=(136, original number of video frames)
    return -- numpy array, dtype=np.float32, shape=(136, original number of video frames)
    ''' 
    vector_copy = np.copy(vector)
    for frame in range(1,vector.shape[1]):
        vector[:,frame] -= vector_copy[:,frame-1]
    vector[:,0] = np.zeros((vector.shape[0]),dtype=np.float32)
    return vector

def resample_video_vector(vector,target_length):
    '''upsample video vector to target_length
    Keywords arguments:
    vector        -- numpy array, dtype=np.float32, shape=(136, original number of video frames)
    target_length -- time frames of spectrogram
    return        -- numpy array, dtype=np.float32, shape=(136, time frames of spectrogram)
    ''' 
    idx_of_new = 0
    idx_of_old = 0
    new_vector = np.zeros((vector.shape[0],target_length),dtype=np.float32)
    while idx_of_old < vector.shape[1]:
        if (idx_of_old+1) % 3 == 0:
            new_vector[:,idx_of_new:idx_of_new+4] = np.array([vector[:,idx_of_old],]*4).transpose()
            idx_of_new += 4
        else:
            new_vector[:,idx_of_new:idx_of_new+3] = np.array([vector[:,idx_of_old],]*3).transpose()
            idx_of_new += 3
        idx_of_old += 1
    return new_vector

def normalize_video_vector(vector):
    '''range value of video vector to [-1,1]

    '''
    max_abs_scaler = preprocessing.MaxAbsScaler()
    vector = max_abs_scaler.fit_transform(vector)
    return vector

def spectral_convergence(input, target):
  return 20 * ((input - target).norm().log10() - target.norm().log10())

def melspecfunc(waveform):
  specgram = specfunc(waveform)
  mel_specgram = melfunc(specgram)
  return mel_specgram

def spec_to_audio(spec, transform_fn, samples=None, init_x0=None, maxiter=1000, tol=1e-6, verbose=1, evaiter=10, lr=0.003):
    '''
    Grinffin-Lim algorithm
    '''
    spec = torch.Tensor(spec)
    samples = (spec.shape[-1]*hypara.hop_size)-hypara.hop_size

    if init_x0 is None:
        init_x0 = spec.new_empty((1,samples)).normal_(std=1e-6)
    x = nn.Parameter(init_x0)
    T = spec

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam([x], lr=lr)

    bar_dict = {}
    metric_func = spectral_convergence
    bar_dict['spectral_convergence'] = 0
    metric = 'spectral_convergence'

    init_loss = None
    with tqdm(total=maxiter, disable=not verbose) as pbar:
        for i in range(maxiter):
            optimizer.zero_grad()
            V = transform_fn(x)
            loss = criterion(V, T)
            loss.backward()
            optimizer.step()
            lr = lr*0.9999
            for param_group in optimizer.param_groups:
              param_group['lr'] = lr

            if i % evaiter == evaiter - 1:
                with torch.no_grad():
                    V = transform_fn(x)
                    bar_dict[metric] = metric_func(V, spec).item()
                    l2_loss = criterion(V, spec).item()
                    pbar.set_postfix(**bar_dict, loss=l2_loss)
                    pbar.update(evaiter)
    return np.array(x.detach().view(-1).cpu())

def create_dataset(audio_path='dataset/audio/train',video_path='dataset/video/train'):
  ''' create tensorflow dataset
    Parameters
    ----------
    audio_path : str
        audio data path.
    video_path : str
        video data path.

    Returns
    -------
    data : tensorflow Dataset
        (x,label).

    '''
  spec = load_mel_array(audio_path)
  spec_crpt = corruption_spec(spec,start_frame=hypara.start_frame,end_frame=hypara.end_frame)
  video = load_video_array(path=video_path) 
  mix_crpt = np.concatenate((spec_crpt,video),axis=1) #(num of sample,feature,time)
  del spec_crpt, video
  AV_Y = Dataset.from_tensor_slices(spec.transpose((0,2,1)))            # 原av特征图(训练)
  AV_X = Dataset.from_tensor_slices(mix_crpt.transpose((0,2,1)))        # 损坏av特征图图(训练)
  data = Dataset.zip((AV_X, AV_Y)).batch(hypara.batch_size)
  return data
    

if __name__ == '__main__':
  spec_data = load_mel_array('dataset/audio/train/')
  c_spec_data = corruption_spec(spec_data)
  
  video_data = load_video_array(path='dataset/video/train')
  mix_data = np.concatenate((c_spec_data,video_data),axis=1)
  del video_data
