import tensorflow as tf
import torch
import numpy as np
from data_loader import *
from Config import Configuration
from network import model_Seq, mymodel
from tensorflow.data import Dataset
import soundfile as sf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, save_model
from librosa.display import specshow
import datetime
import sounddevice as sd
hypara = Configuration()

def plot_mel(spec_est,spec):
  fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
  img0 = specshow(spec_est,y_axis='mel',x_axis='time',ax=ax[0])
  img1 = specshow(spec,y_axis='mel',x_axis='time',ax=ax[1])
  ax[0].set(title='inpainted mel-spectrogram')
  ax[1].set(title='true mel-spectrogram')
  fig.colorbar(img1, ax=ax[1], format="%+2.2f dB")
  plt.show()
  
def wv_array(path='dataset/audio/test/s30'):
  '''
  Load audio from individual folder, like dataset/audio/test/s30
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

def plot_loss(history):
  x_axis = range(0,hypara.epochs)
  loss = history.history['loss']
  val_loss= history.history['val_loss']
  plt.plot(x_axis, loss, 'r', label='Training MSE')
  plt.legend(loc='upper right')
  plt.plot(x_axis, val_loss, 'b', label='Validation MSE')
  plt.legend(loc='upper right')
  plt.ylabel('Loss Value')
  plt.xlabel('Epoch')
  plt.title('Loss')
  plt.show()

    

def train(model,train_data,val_data,load=False):
  if load:
      model.load_weights('models/Seqmodel_mse_0.01331643108278513.h5')
  epochs = hypara.epochs
  log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  
  history = model.fit(train_data, batch_size=hypara.batch_size,\
                      epochs=epochs,validation_data=val_data,callbacks=[tensorboard_callback])
  plot_loss(history)
  return model

def mse(x1,x2):
    row,col = x1.shape
    total_num = row * col
    res = 0
    for r in range(row):
        for c in range(col):
            res += (x1[r][c]-x2[r][c]) ** 2 / total_num
    return res


if __name__ == '__main__':
  
  # train model
  train_data = create_dataset('dataset/audio/train','dataset/video/train')
  val_data = create_dataset('dataset/audio/val','dataset/video/val')
  model = train(model_Seq,train_data,val_data,load=True)
  del train_data,val_data
  
  
  
  # test model
  num_of_test = 10
  test_data = create_dataset('dataset/audio/test','dataset/video/test')
  print('start audio inpainting ...')
  test_spec_est = model.predict(test_data).transpose((0,2,1))[:num_of_test]
  print('audio inpainting completed!')
    
  mse_loss = model.evaluate(test_data)
  print(f'mse is {mse_loss}')
  model_path = f'models/Seqmodel_mse_{mse_loss}.h5'
  model.save(model_path)
  test_spec_true = load_mel_array('dataset/audio/test/')[:num_of_test]
 
  plot_mel(test_spec_est[5],test_spec_true[5])
  test_spec_est = librosa.db_to_power(denormalize_melspec(test_spec_est) + hypara.ref_level_db)
  test_spec_true = librosa.db_to_power(denormalize_melspec(test_spec_true) + hypara.ref_level_db)
  wv_true = wv_array()[:num_of_test]
  
  start_frame = hypara.start_frame
  end_frame = hypara.end_frame
  hop = hypara.hop_size
  sr = hypara.sample_rate
  for i in range(num_of_test):
      wv_crpt = np.copy(wv_true[i])
      wv_crpt[start_frame*hop:end_frame*hop] = -np.ones(((end_frame-start_frame)*hop))
      wv_est = librosa.util.normalize(spec_to_audio(np.expand_dims(test_spec_est[i],0), melspecfunc, maxiter=500, evaiter=10, tol=1e-8))
      
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
        
      

  

  
