

from __future__ import print_function
from __future__ import absolute_import


import tensorflow as tf



class EncoderBase(object):
    """
    Base class for a encoder, which encode a sample to a latent variable
    """
    def __init__(self,
                 input_dim,
                 encoded_dim,
                 encoded_desc,
                 name_scope,
                 weight_decay=2e-5,
                 dtype=tf.float32):
        self.input_dim = input_dim
        self.encoded_dim = encoded_dim
        self.encoded_desc = encoded_desc
        self.name_scope = name_scope
        self.weight_decay=weight_decay
        self.dtype = dtype
        

    def forward(self, encoder_inputs, trainable=True, is_training=True):
        raise NotImplementedError("The forward method must be implemented: class %s" % type(self).__name__)


class DecoderBase(object):
    """
    Base class for a decoder, which decode a latent variable to a sample. This can also be the generator in an adversirial network.
    """
    def __init__(self,
                 input_dim,
                 category_dim,
                 decoded_dim,
                 name_scope,
                 weight_decay=2e-5,
                 dtype=tf.float32):
        self.input_dim = input_dim
        self.category_dim = category_dim
        self.decoded_dim = decoded_dim
        self.name_scope = name_scope
        self.weight_decay=weight_decay
        self.dtype = dtype

        

    def forward(self, decoder_hidden, decoder_category, trainable=True, is_training=True):
        raise NotImplementedError("The forward method must be implemented: class" % type(self).__name__)


class DiscriminatorBase(object):
    """
    Base class for a discriminator in an adversirial network.
    """
    def __init__(self, input_dim,
                 class_num,
                 name_scope,
                 weight_decay=2e-5,
                 dtype=tf.float32):
        self.input_dim = input_dim
        self.class_num = class_num
        self.name_scope = name_scope
        self.dtype = dtype
        self.weight_decay=weight_decay

        
    def forward(self, data_input, input_class, 
                reuse=False, trainable=True, is_training=True):
        raise NotImplementedError("The forward method must be implemented: class" % type(self).__name__)
