import VAE

import tensorflow as tf
import tensorflow.contrib.layers as tcl

import os

import math
from tensorflow.python.ops import math_ops


class ACEncoder(VAE.EncoderBase):
    def __init__(self, nfilters, re_term,
                 enc_dim=527,
                 enc_dim_desc={'hidden_num': 512, 'class_num': 15},
                 enc_shape=[None, 49, 54, 1], name_scope='EncoderSkeleton'):
        super(ACEncoder, self).__init__(enc_shape,
                                        enc_dim,
                                        enc_dim_desc,
                                        name_scope)
        self.nfilters = nfilters
        self.re_term = re_term

    def forward(self, encoder_inputs, trainable=True, is_training=True, reuse=False, with_batchnorm=False):
        with tf.variable_scope(self.name_scope, reuse=reuse) as vs:
            if (reuse):
                vs.reuse_variables()
            lrelu = VAE.lrelu

            if (with_batchnorm):
                print('here')
                h0 = lrelu(tcl.batch_norm(tcl.conv2d(encoder_inputs,
                                                     num_outputs=self.nfilters * 4,
                                                     stride=2,
                                                     kernel_size=[2, 7],
                                                     activation_fn=None,
                                                     padding='SAME',
                                                     biases_initializer=None,
                                                     weights_regularizer=tcl.l2_regularizer(self.re_term),
                                                     scope="conv1"),
                                          scope='bn1',
                                          trainable=trainable,
                                          is_training=is_training))

                h0 = lrelu(tcl.batch_norm(tcl.conv2d(h0,
                                                     num_outputs=self.nfilters * 4,
                                                     stride=2,
                                                     kernel_size=[2, 7],
                                                     activation_fn=None,
                                                     padding='SAME',
                                                     scope="conv2",
                                                     weights_regularizer=tcl.l2_regularizer(self.re_term),
                                                     biases_initializer=None),
                                          trainable=trainable,
                                          scope='bn2',
                                          is_training=is_training))

                h0 = tcl.dropout(h0, 0.8, is_training=is_training)

                h0 = lrelu(tcl.batch_norm(tcl.conv2d(h0,
                                                     num_outputs=self.nfilters * 8,
                                                     stride=2,
                                                     kernel_size=[2, 7],
                                                     activation_fn=None,
                                                     padding='SAME',
                                                     scope="conv3",
                                                     weights_regularizer=tcl.l2_regularizer(self.re_term),
                                                     biases_initializer=None),
                                          trainable=trainable,
                                          scope='bn3',
                                          is_training=is_training))
            else:
                h0 = lrelu(tcl.conv2d(encoder_inputs,
                                      num_outputs=self.nfilters * 4,
                                      stride=2,
                                      kernel_size=[2, 7],
                                      activation_fn=None,
                                      padding='SAME',
                                      biases_initializer=None,
                                      weights_regularizer=tcl.l2_regularizer(self.re_term),
                                      scope="conv1"))
                h0 = tcl.dropout(h0, 0.8, is_training=is_training)
                h0 = lrelu(tcl.conv2d(h0,
                                      num_outputs=self.nfilters * 8,
                                      stride=2,
                                      kernel_size=[2, 7],
                                      activation_fn=None,
                                      padding='SAME',
                                      scope="conv2",
                                      weights_regularizer=tcl.l2_regularizer(self.re_term),
                                      biases_initializer=None))
                h0 = tcl.dropout(h0, 0.8, is_training=is_training)

                h0 = lrelu(tcl.conv2d(h0,
                                      num_outputs=self.nfilters * 8,
                                      stride=2,
                                      kernel_size=[2, 7],
                                      activation_fn=None,
                                      padding='SAME',
                                      scope="conv3",
                                      weights_regularizer=tcl.l2_regularizer(self.re_term),
                                      biases_initializer=None))
                h0 = tcl.dropout(h0, 0.5, is_training=is_training)

            h0 = tcl.flatten(h0)

            h0 = tcl.fully_connected(h0, self.encoded_dim,
                                     weights_regularizer=tcl.l2_regularizer(self.re_term),
                                     scope="fc1", activation_fn=None)

            return h0


class AEDecoder(VAE.DecoderBase):
    def __init__(self, nfilters, re_term):
        super(AEDecoder, self).__init__([None, 128], [None, 15], [None, 1, 54, 1], 'DecoderSkeleton')
        self.nfilters = nfilters
        self.re_term = re_term
        self.encoder = ACEncoder(nfilters, self.re_term,
                                 enc_shape=[None, 20, 54, 1],
                                 enc_dim=512,
                                 enc_dim_desc={'hidden_num': 512},
                                 name_scope='DecoderSkeletonEnc')
    def forward(self, dec_in, reuse=False, trainable=True, is_training=True):
        with tf.variable_scope(self.name_scope) as vs:
            if (reuse):
                vs.reuse_variables()
            lrelu = VAE.lrelu

            dec_in_enc = self.encoder.forward(dec_in, reuse=reuse, trainable=trainable, is_training=is_training)


            h0 = tcl.fully_connected(dec_in_enc, 512, scope="fc3", activation_fn=lrelu,
                                     weights_regularizer=tcl.l2_regularizer(self.re_term))

            h0 = tcl.dropout(h0, 0.5, is_training=is_training)

            h0 = tcl.fully_connected(h0, 54, scope="fc4", activation_fn=None,
                                     weights_regularizer=tcl.l2_regularizer(self.re_term), )

            h0 = tf.expand_dims(tf.expand_dims(h0, 1), 3)

            return h0

class Discriminator(VAE.DiscriminatorBase):
    def __init__(self, nfilters, re_term):
        super(Discriminator, self).__init__([None, 75, 54, 1],
                                            15,
                                            "DiscriminatorSkeleton")
        self.nfilters = nfilters
        self.encoder = ACEncoder(nfilters, re_term,
                                 enc_shape=[None, 75, 54, 1],
                                 enc_dim=512,
                                 enc_dim_desc={'hidden_num': 512},
                                 name_scope='DiscriminatorSkeletonEnc')
        self.re_term = re_term

    def forward(self, data_input, input_class,
                reuse=False, trainable=True, is_training=True):
        with tf.variable_scope(self.name_scope) as vs:
            if (reuse):
                vs.reuse_variables()

            dec_in_enc = self.encoder.forward(data_input, reuse=reuse, trainable=trainable, is_training=is_training,
                                              with_batchnorm=False)

            dec_in_enc = tf.nn.relu(dec_in_enc)

            y = tf.concat([dec_in_enc, input_class], 1)

            h0 = tcl.fully_connected(y, self.nfilters * 8, scope="fc1",
                                     weights_regularizer=tcl.l2_regularizer(self.re_term))

            return tcl.fully_connected(h0, 1, activation_fn=None, weights_regularizer=tcl.l2_regularizer(self.re_term))

