import numpy as np
import os
import tensorflow as tf
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput


params = {"learning_rate": LEARNING_RATE}

def model_fn(features, labels, mode, params):
    


def serving_input_fn(params):
    




def train_input_fn(training_dir, params):



def eval_input_fn(training_dir, params):



def _input_fn(training_dir, training_filename):
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=os.path.join(training_dir, training_filename), target_dtype=np.int, features_dtype=np.float32)

    return tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)()