# Import this file to make script work with GPU!
# import it first before importing tensorflow!!!

# JL - CUDA PATCH
import os


# print(tf.config.list_physical_devices('GPU'))
# https://stackoverflow.com/questions/75968226/how-can-i-install-tensorflow-and-cuda-drivers

def setup_cuda():
    os.environ['PATH'] = os.environ[
                             'PATH'] + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin;C:\\Program Files\\NVIDIA\\CUDNN\\v8.9.7\\bin"
    os.environ['CUDA_HOME'] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit"
    os.environ[
        'LD_LIBRARY_PATH'] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\lib;C:\\Program Files\\NVIDIA\\CUDNN\\v8.9.7\\lib"

setup_cuda()


import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    # logical_gpus = tf.config.list_logical_devices('GPU')
    # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    raise e

print("GPU List: " + str(tf.config.list_physical_devices('GPU')))