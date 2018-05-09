# coding='utf-8'
'''
    author: Youzhao Yang
    date: 05/08/2018
    github: https://github.com/nnuyi
'''

import tensorflow as tf
import numpy as np
import time
import os

from tqdm import tqdm
from layers import Layer
from utils import get_data, gen_batch_data

class PeleeNet:
    model_name = 'PeleeNet'
    '''
        PeleeNet Class
    '''
    def __init__(self, config=None, sess=None):
        self.sess = sess
        self.config = config
        
        self.num_class = self.config.num_class
        self.input_height = self.config.input_height
        self.input_width = self.config.input_width
        self.input_channel = self.config.input_channel
        
        self.batchsize = self.config.batchsize
        
        self.layer = Layer()
        
    def peleenet(self, input_x, k=32, num_init_channel=64, block_config=[3,4,8,6], bottleneck_width=[2,2,4,4], is_training=True, reuse=False):
        with tf.variable_scope(self.model_name) as scope:
            if reuse:
                scope.reuse_variables()
                
            '''
            --------------------------------------------------------------------
                                    feature extraction
            --------------------------------------------------------------------
            '''
            # _stem_block(self, input_x, num_init_channel=32, is_training=True, reuse=False):    
            from_layer =  self.layer._stem_block(input_x,
                                                 num_init_channel=num_init_channel,
                                                 is_training=is_training,
                                                 reuse=reuse)
            
            # _dense_block(self, input_x, stage, num_block, k, bottleneck_width, is_training=True, reuse=False):
            # _transition_layer(self, input_x, stage, is_avgpool=True, is_training=True, reuse=False):
            stage = 0
            for num_block, bottleneck_coeff in zip(block_config, bottleneck_width):
                stage = stage + 1
                # dense_block
                from_layer = self.layer._dense_block(from_layer,
                                                     stage,
                                                     num_block,
                                                     k,
                                                     bottleneck_coeff,
                                                     is_training=is_training,
                                                     reuse=reuse)

                is_avgpool = True if stage < 4 else False
                output_channel = from_layer.get_shape().as_list()[-1]
                # transition_layer
                from_layer = self.layer._transition_layer(from_layer,
                                                          stage,
                                                          output_channel=output_channel,
                                                          is_avgpool=is_avgpool,
                                                          is_training=is_training,
                                                          reuse=reuse)

            '''
            --------------------------------------------------------------------
                                    classification
            --------------------------------------------------------------------
            '''
            # _classification_layer(self, input_x, num_class, keep_prob=0.5, is_training=True, reuse=False):
            logits = self.layer._classification_layer(from_layer, self.num_class, is_training=is_training, reuse=reuse)
            return logits

    def build_model(self):
        self.input_train = tf.placeholder(tf.float32, [self.batchsize, self.input_height, self.input_width, self.input_channel], name='input_train')
        self.input_test = tf.placeholder(tf.float32, [self.batchsize, self.input_height, self.input_width, self.input_channel], name='input_test')
        self.one_hot_labels = tf.placeholder(tf.float32, [self.batchsize, self.num_class], name='one_hot_labels')

        # logits data and one_hot_labels
        self.logits_train = self.peleenet(self.input_train, is_training=True, reuse=False)
        self.logits_test = self.peleenet(self.input_test, is_training=False, reuse=True)
        # self.one_hot_labels = tf.one_hot(self.input_label, self.num_class)
        
        # loss function
        def softmax_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.softmax_cross_entropy_with_logits(targets=x, labels=y)
        # weights regularization
        self.weights_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = tf.reduce_mean(softmax_cross_entropy_with_logits(self.logits_train, self.one_hot_labels)) + self.config.weight_decay*self.weights_reg
        
        # optimizer
        '''
        self.adam_optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate,
                                                 beta1=self.config.beta1,
                                                 beta2=self.config.beta2).minimize(self.loss)
        '''
        self.rmsprop_optim = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate,
                                                       momentum=self.config.momentum).minimize(self.loss)

        # accuracy
        self.predicetion = tf.nn.softmax(self.logits_test, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predicetion, 1), tf.argmax(self.one_hot_labels, 1)), tf.float32))
        
        # summary
        self.loss_summary = tf.summary.scalar('entrophy loss', self.loss)
        self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('logs', self.sess.graph)
        
        # saver
        self.saver = tf.train.Saver()
        
    def train_model(self):
        # initialize variables
        tf.global_variables_initializer().run()
        
        # load model
        if self.load_model():
            print('load model successfully')
        else:
            print('fail to load model')
            
        # get datasource
        datasource = get_data(self.config.dataset, is_training=True)
        gen_data = gen_batch_data(datasource, self.batchsize, is_training=True)
        ites_per_epoch = int(len(datasource.images)/self.batchsize)
        
        step = 0
        for epoch in range(self.config.epochs):
            for ite in tqdm(range(ites_per_epoch)):
                images, labels = next(gen_data)
                _, loss, accuracy, summaries = self.sess.run([self.rmsprop_optim, self.loss, self.accuracy, self.summaries], feed_dict={
                                                                                        self.input_train:images,
                                                                                        self.input_test:images,
                                                                                        self.one_hot_labels:labels
                                                                                        })
                
                step = step + 1
                self.summary_writer.add_summary(summaries, global_step=step)
            
            # test model
            if np.mod(epoch, 1) == 0:
                print('--epoch_{} -- training accuracy:{}'.format(epoch, accuracy))
                self.test_model()

            # save model
            if np.mod(epoch, 5) == 0:
                self.save_model()
        
    def test_model(self):
        if not self.config.is_training:
            # initialize variables
            tf.global_variables_initializer().run()
            # load model
            if self.load_model():
                print('load model successfully')
            else:
                print('fail to load model')
        
        datasource = get_data(self.config.dataset, is_training=False)      
        gen_data = gen_batch_data(datasource, self.batchsize, is_training=False)
        ites_per_epoch = int(len(datasource.images)/self.batchsize)
        
        accuracy = []
        for ite in range(ites_per_epoch):
            images, labels = next(gen_data)
            accuracy_per_epoch = self.sess.run([self.accuracy], feed_dict={
                                                                            self.input_test:images,
                                                                            self.one_hot_labels:labels
                                                                            })
            accuracy.append(accuracy_per_epoch[0])
    
        acc = np.mean(accuracy)
        print('--test epoch -- accuracy:{:.4f}'.format(acc))
        
    # load model
    def load_model(self):
        if not os.path.isfile(os.path.join(self.model_dir, 'checkpoint')):
            return False
        self.save.restore(self.sess, self.model_pos)
    
    # save model
    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        self.saver.save(self.sess, self.model_pos)
    
    @property
    def model_dir(self):
        return '{}/{}'.format(self.config.checkpoint_dir, self.config.dataset)
    
    @property
    def model_pos(self):
        return '{}/{}/{}'.format(self.config.checkpoint_dir, self.config.dataset, self.model_name)

if __name__=='__main__':
    input_x = tf.placeholder(tf.float32, [64, 224,224,3], name='input_train')
    peleenet = PeleeNet()
    start_time = time.time()
    output = peleenet.peleenet(input_x)
    end_time = time.time()
    print('total time:{}'.format(end_time-start_time))
    print(output.get_shape().as_list())
