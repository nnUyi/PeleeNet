# coding='utf-8'
'''
    author: Youzhao Yang
    date: 05/08/2018
    github: https://github.com/nnuyi
'''

import tensorflow as tf
from config import Config
from PeleeNet import PeleeNet

def main():
    config = Config()
    config.check_dir()
    config.print_config()
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    gpu_options.allow_growth = True
    
    sess_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    with tf.Session(config=sess_config) as sess:
        peleenet = PeleeNet(config=config.config, sess=sess)
        peleenet.build_model()
        if config.config.is_training:
            peleenet.train_model()
        if config.config.is_testing:
            peleenet.test_model()

if __name__=='__main__':
    main()
