import tensorflow as tf
import numpy as np
from data_loader import audio_array, audio_to_spec, corruption_spec
from Config import Configuration

hypara = Configuration()
root_path = hypara.root_path




def train(model,data):



    return


if __name__ == '__main__':
  
  train_wv = audio_array(root_path+'dataset/audio/train')
  train_spec = audio_to_spec(train_wv)
  train_spec_crpt = corruption_spec(train_spec)
  
  val_wv = audio_array(root_path+'dataset//audio/val')
  val_spec = audio_to_spec(val_wv)
  val_spec_crpt = corruption_spec(val_spec)

  train_A_Y = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(train_spec)).batch(hypara.batch_size, drop_remainder=True)     # 原频谱图(训练)
  train_A_X = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(train_spec_crpt)).batch(hypara.batch_size, drop_remainder=True)# 损坏频谱图(训练)

  val_A_Y = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(val_spec)).batch(hypara.batch_size, drop_remainder=True)         # 原频谱图(验证)
  val_A_X = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(val_spec)).batch(hypara.batch_size, drop_remainder=True)         # 损坏频谱图(验证)
  
  



'''
  train_A_Y # 原频谱图(训练)
  train_A_X # 损坏频谱图(训练)
  train_V   # 视频向量(训练)

  val_A_Y # 原频谱图(验证)
  val_A_X # 损坏频谱(验证)
  val_V   # 视频向量(验证)

  test_A_Y # 原频谱图(测试)
  test_A_X # 损坏频谱(测试)
  test_V   # 视频向量(测试)
'''

  
