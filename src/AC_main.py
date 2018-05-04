#!/usr/bin/env python

import VAE


import importlib
import humanEncoder
import humanEncoder_cmu
import humanEncoder_ablation
import importlib
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y-', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parse = argparse.ArgumentParser()

parse.add_argument("--l2regWeight", help="the weight of l2 regularizer", default=0.001, type=float)
parse.add_argument("--gan_loss_weight", help="the weight of gan loss", default=0.01, type=float)
parse.add_argument("--batch_size", help="the batch size used in trainning", default=16, type=int)
parse.add_argument("--use_sampling", help="use predict resutls in training instead of ground truth", type=float, default=0.95)
parse.add_argument("--window_length",help="how many previous frames should be used to predict the current frame",type=int,default=20)
parse.add_argument("--output_length",help="how many previous frames should be used to predict the current frame",type=int,default=25)
parse.add_argument("--is_sampling",help="the current phase",type=bool,default=False)
parse.add_argument("--checkpoint", help="specify which model to load in", default=20000, type=int)
parse.add_argument("--dataset",help="choose a dataset for training",default="human3.6m",type=str)
args = parse.parse_args()

print(args)

re_term = args.l2regWeight
if args.dataset=="human3.6m":

    if not args.is_sampling:
        dloader = VAE.DataLoader(50, 25, './data/h3.6m/dataset')
    else:
        dloader = VAE.DataLoader(50, args.output_length, './data/h3.6m/dataset')
    encoder = humanEncoder.ACEncoder(16, re_term)
    decoder = humanEncoder.AEDecoder(16, re_term)
    discriminator = humanEncoder.Discriminator(32, re_term)
    inputDimension=54

    ###---------train on the cmu dataset,the difference is that the input dimension is 70 ------------------
else:
    if not args.is_sampling:
        dloader=VAE.Data_cmu(50,25,'./data/cmu_mocap/train','./data/cmu_mocap/test')
    else:
        dloader = VAE.Data_cmu(50, args.output_length, './data/cmu_mocap/train','./data/cmu_mocap/test')
    encoder = humanEncoder_cmu.ACEncoder(16, re_term)
    decoder = humanEncoder_cmu.AEDecoder(16, re_term)
    discriminator = humanEncoder_cmu.Discriminator(32, re_term)
    inputDimension = 70


###---------for ablation study, remove the long term CEM from our model ------------------
# encoder = humanEncoder_ablation.ACEncoder(16, re_term)
# decoder = humanEncoder_ablation.AEDecoder(16, re_term)
# discriminator = humanEncoder_ablation.Discriminator(32, re_term)

gan_loss_weight=args.gan_loss_weight

batch_size = args.batch_size

model_name = 'CNNAdTrain_GANWEIGHT%f_Sampling%fWindownLength%d' % (args.gan_loss_weight, args.use_sampling, args.window_length)

VAETrain = VAE.Autoencoder_gan.Autoenc_gan(encoder,
                                           decoder,
                                           discriminator,
                                           [None, 75, inputDimension, 1] if not args.is_sampling else [None,50+args.output_length,70,1],
                                           50,
                                           25 if not args.is_sampling else args.output_length,
                                           modelname=model_name,
                                           learning_rate=5e-5,
                                           batch_size=batch_size,
                                           gan_loss_weight=gan_loss_weight,
                                           sampling=args.use_sampling,
                                           window_length=args.window_length,
                                           concat_input_output=True,
                                           trainable=True if not args.is_sampling else False,
                                           is_sampling=args.is_sampling,
                                           dataset_name=args.dataset)

with tf.Session() as sess:
    if not args.is_sampling:
        sess.run(tf.global_variables_initializer())
        VAETrain.Train(sess, dloader)
    else:
        # VAETrain.Saver.restore(sess,'./Models/Models_kernel54/{}-{}'.format(model_name,args.checkpoint))
        VAETrain.Saver.restore(sess, './Models/{}-{}'.format(model_name, args.checkpoint))
        VAETrain.InferenceSample(sess,dloader)
