import tensorflow as tf


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)
