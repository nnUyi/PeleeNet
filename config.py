# coding='utf-8'
'''
    author: Youzhao Yang
    date: 05/08/2018
    github: https://github.com/nnuyi
'''

import tensorflow as tf
import os

class Config:
    def __init__(self):
        self.flags = tf.app.flags
        self.flags.DEFINE_integer('epochs', 500, 'training epochs')
        self.flags.DEFINE_integer('batchsize', 64, 'training batchsize')
        self.flags.DEFINE_integer('input_height', 32, 'input height')
        self.flags.DEFINE_integer('input_width', 32, 'input width')
        self.flags.DEFINE_integer('input_channel', 3, 'input channel')
        self.flags.DEFINE_integer('num_class', 10, 'numbers of class')
        self.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
        self.flags.DEFINE_float('beta1', 0.9, 'beta1')
        self.flags.DEFINE_float('beta2', 0.999, 'beta2')
        self.flags.DEFINE_float('momentum', 0.9, 'monument for rmsprop optimizer')
        self.flags.DEFINE_float('weight_decay', 5e-4, 'weight decay')
        self.flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'checkpoint directory')
        self.flags.DEFINE_string('logs_dir', 'logs', 'logs directory')
        self.flags.DEFINE_string('dataset', 'cifar10', 'dataset type')
        self.flags.DEFINE_bool('is_training', False, 'training or testing')
        self.flags.DEFINE_bool('is_testing', False, 'training or testing')

        self.config = self.flags.FLAGS

    def check_dir(self):
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)
        if not os.path.exists(self.config.logs_dir):
            os.mkdir(self.config.logs_dir)

    def print_config(self):
        print('Config Proto:')
        print('-'*30)
        print('dataset:{}'.format(self.config.dataset))
        print('epochs:{}'.format(self.config.epochs))
        print('batchsize:{}'.format(self.config.batchsize))
        print('learning_rate:{}'.format(self.config.learning_rate))
        print('beta1:{}'.format(self.config.beta1))
        print('beta2:{}'.format(self.config.beta2))
        print('-'*30)
