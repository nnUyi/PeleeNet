# coding='utf-8'
'''
    author: Youzhao Yang
    date: 05/08/2018
    github: https://github.com/nnuyi
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim

class Layer:        
    # stem_block       
    def _stem_block(self, input_x, num_init_channel=32, is_training=True, reuse=False):
        block_name = 'stem_block'
        with tf.variable_scope(block_name) as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               normalizer_fn=slim.batch_norm,
                                               activation_fn=tf.nn.relu) as s:
                conv0 = slim.conv2d(input_x, num_init_channel, 3, 1, scope='stem_block_conv0')
                
                conv1_l0 = slim.conv2d(conv0, int(num_init_channel/2), 1, 1, scope='stem_block_conv1_l0')
                conv1_l1 = slim.conv2d(conv1_l0, num_init_channel, 3, 1, scope='stem_block_conv1_l1')
                
                maxpool1_r0 = slim.max_pool2d(conv0, 2, 1, padding='SAME', scope='stem_block_maxpool1_r0')
                
                filter_concat = tf.concat([conv1_l1, maxpool1_r0], axis=-1)
                
                output = slim.conv2d(filter_concat, num_init_channel, 1, 1, scope='stem_block_output')
                
            return output

    def _dense_block(self, input_x, stage, num_block, k, bottleneck_width, is_training=True, reuse=False):
        with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           normalizer_fn=slim.batch_norm,
                                           activation_fn=tf.nn.relu) as s:
            output = input_x
            
            for index in range(num_block):
                dense_block_name = 'stage_{}_dense_block_{}'.format(stage, index)
                with tf.variable_scope(dense_block_name) as scope:
                    if reuse:
                        scope.reuse_variables()
                    
                    inter_channel = k*bottleneck_width
                    # left channel
                    conv_left_0 = slim.conv2d(output, inter_channel, 1, 1, scope='conv_left_0')
                    conv_left_1 = slim.conv2d(conv_left_0, k, 3, 1, scope='conv_left_1')
                    # right channel
                    conv_right_0 = slim.conv2d(output, inter_channel, 1, 1, scope='conv_right_0')
                    conv_right_1 = slim.conv2d(conv_right_0, k, 3, 1, scope='conv_right_1')
                    conv_right_2 = slim.conv2d(conv_right_1, k, 3, 1, scope='conv_right_2')
                    
                    output = tf.concat([output, conv_left_1, conv_right_2], axis=3)
            return output
                    
    def _transition_layer(self, input_x, stage, output_channel, is_avgpool=True, is_training=True, reuse=False):
        transition_layer_name = 'stage_{}_transition_layer'.format(stage)
        
        with tf.variable_scope(transition_layer_name) as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               normalizer_fn=slim.batch_norm,
                                               activation_fn=tf.nn.relu) as s:
                conv0 = slim.conv2d(input_x, output_channel, 1, 1, scope='transition_layer_conv0')
                if is_avgpool:
                    output = slim.avg_pool2d(conv0, 2, 2, scope='transition_layer_avgpool')
                else:
                    output = conv0
            return output
    
    def _classification_layer(self, input_x, num_class, keep_prob=0.5, is_training=True, reuse=False):
        classification_layer_name = 'classification_layer'
        with tf.variable_scope(classification_layer_name) as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.fully_connected], weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                        normalizer_fn=None,
                                                        activation_fn=None), \
                 slim.arg_scope([slim.dropout], keep_prob=keep_prob) as s:
                
                shape = input_x.get_shape().as_list()
                filter_size = [shape[1], shape[2]]
                global_avgpool = slim.avg_pool2d(input_x, filter_size, scope='global_avgpool')
                
                # dropout
                # dropout = slim.dropout(global_avgpool)
                flatten = tf.reshape(global_avgpool, [shape[0], -1])
                logits = slim.fully_connected(flatten, num_class, scope='fc')
                
                return logits
    
if __name__=='__main__':
    input_x = tf.Variable(tf.random_normal([64,224,224,32]))
    layer = Layer()
    stem_block_output = layer._stem_block(input_x, 32)
    dense_block_output = layer._dense_block(input_x, 0, 3, 16, 2)
    transition_layer_output = layer._transition_layer(dense_block_output, 0, is_avgpool=False)
    print(stem_block_output.get_shape().as_list())
    print(dense_block_output.get_shape().as_list())
    print(transition_layer_output.get_shape().as_list())
