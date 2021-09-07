import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Activation, GRU, Dropout
from Config import Configuration
import numpy as np
hypara = Configuration()

start = hypara.start_frame
end = hypara.end_frame



class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    layers = [256,128,64,128]
    self.BLSTM1 = tf.keras.layers.Bidirectional(LSTM(layers[0], return_sequences=True),input_shape=(251,hypara.n_mels+136),merge_mode='concat')
    self.droput1 = Dropout(0.2)
    self.BLSTM2 = tf.keras.layers.Bidirectional(LSTM(layers[1], return_sequences=True),merge_mode='concat')
    self.droput2 = Dropout(0.2)
    self.BLSTM3 = tf.keras.layers.Bidirectional(LSTM(layers[2], return_sequences=True),merge_mode='concat')
    self.droput3 = Dropout(0.2)
    self.FC = tf.keras.layers.Dense(layers[3],input_shape=(251,hypara.n_mels), activation='tanh')
  def call(self, inputs, training=False):
    #start = np.random.randint(low=80,high=150)
    if training:
        start = np.random.randint(low=80,high=150)
        end = start + 50
    else:
        start = hypara.start_frame
        end = hypara.end_frame
    mask_for_valid_inputs = tf.zeros(tf.shape(inputs),dtype=np.float32)    
    mask_for_valid_inputs_list = tf.unstack(mask_for_valid_inputs,axis=1)
    mask_for_surround_inputs = tf.ones(tf.shape(inputs),dtype=np.float32)
    mask_for_surround_inputs_list = tf.unstack(mask_for_surround_inputs,axis=1)
    for i in range(start,end):
        mask_for_valid_inputs_list[i] = -2 * tf.ones(tf.shape(mask_for_valid_inputs_list[i]),dtype=np.float32)
        mask_for_surround_inputs_list[i] = tf.zeros(tf.shape(mask_for_surround_inputs_list[i]),dtype=np.float32)
    mask_for_valid_inputs = tf.stack(mask_for_valid_inputs_list,axis=1)
    mask_for_surround_inputs = tf.stack(mask_for_surround_inputs_list,axis=1)
    inputs = inputs + mask_for_valid_inputs
    # BLSTM Network
    x = self.BLSTM1(inputs)
    x = self.droput1(x)
    x = self.BLSTM2(x)
    x = self.droput2(x)
    x = self.BLSTM3(x)
    x = self.droput3(x)
    x = self.FC(x)
    # BLSTM Network
    mask_for_out = tf.zeros(tf.shape(x),dtype=np.float32)
    mask_for_out_list = tf.unstack(mask_for_out,axis=1)
    for i in range(start,end):
        mask_for_out_list[i] = tf.ones(tf.shape(mask_for_out_list[i]),dtype=np.float32)
    mask_for_out = tf.stack(mask_for_out_list,axis=1)
    inputs *= mask_for_surround_inputs
    x = x * mask_for_out
    x = x + inputs[:,:,:128]
    return x
mymodel = MyModel() # This network implementation cannot converge for now.
mymodel.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='Adam')

 
model_Seq = Sequential() # Use this one instead
layers = [128,128,128,128]
model_Seq.add(Bidirectional(LSTM(layers[0], return_sequences=True),input_shape=(251,hypara.n_mels+136),merge_mode='concat'))
model_Seq.add(Dropout(0.2)) 
model_Seq.add(Bidirectional(LSTM(layers[1], return_sequences=True),merge_mode='concat'))
model_Seq.add(Dropout(0.2)) 
model_Seq.add(Bidirectional(LSTM(layers[2], return_sequences=True),merge_mode='concat'))
model_Seq.add(Dense(layers[3],input_shape=(251,hypara.n_mels), activation='tanh'))
model_Seq.compile(loss='mse', optimizer='Adam')



