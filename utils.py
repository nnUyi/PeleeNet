# coding='utf-8'
'''
    author: Youzhao Yang
    date: 05/08/2018
    github: https://github.com/nnuyi
'''

import random
import numpy as np
import cifar10
from tensorflow.examples.tutorials.mnist import input_data

from PIL import Image, ImageEnhance, ImageOps, ImageFile

import matplotlib.pyplot as plt
class Datasource:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

def get_data(data_type='mnist', is_training=True):
    if data_type == 'mnist':
        raw_data = input_data.read_data_sets('./data/mnist/', one_hot=True)
        shape = [28,28,1]
        if is_training:
            size = len(raw_data.train.images)
            images = np.reshape(raw_data.train.images, [size]+shape)
            labels = raw_data.train.labels
        else:
            size = len(raw_data.test.images)
            images = np.reshape(raw_data.test.images, [size]+shape)
            labels = raw_data.test.labels
    elif data_type == 'cifar10':
        if is_training:
            images, _, labels = cifar10.load_training_data()
        else:
            images, _, labels = cifar10.load_test_data()
    else:
        raise Exception('data type error: {}'.format(data_type))

    datasource = Datasource(images, labels)
    return datasource

def gen_data(datasource, is_training=True):
    while True:
        indices = list(range(len(datasource.images)))
        random.shuffle(indices)
        if is_training:
            for i in indices:
                image = pre_process(datasource.images[i])
                # image = datasource.images[i]
                label = datasource.labels[i]
                yield image, label
        else:
            for i in indices:
                image = datasource.images[i]
                label = datasource.labels[i]
                yield image, label

def gen_batch_data(datasource, batchsize, is_training=True):
    data_gen = gen_data(datasource, is_training=is_training)
    while True:
        images = []
        labels = []
        for i in range(batchsize):
            image, label = next(data_gen)
            images.append(image)
            labels.append(label)
        yield np.array(images), np.array(labels)

def data_augment(image):
    shape = image.shape
    is_colorful = shape[-1]==3
    # numpy.ndarray to PIL
    if not is_colorful:
        image = Image.fromarray(np.squeeze(np.uint8(image*255)))
    else:
        image = Image.fromarray(np.uint8(image*255))

    def distort_color(image):
        # saturation
        random_factor = np.random.randint(0, 31) / 10.
        color_image = ImageEnhance.Color(image).enhance(random_factor)
        # brightness
        random_factor = np.random.randint(10, 21) / 10.
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        # contrast
        random_factor = np.random.randint(10, 21) / 10.
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        # sharpness
        random_factor = np.random.randint(0, 31) / 10.
        sharpness_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
        
        return sharpness_image

    def distort(image):
        distort_image = image
        # random rotate: angle range from 1 degree to 45 degree
        random_angle = np.random.randint(0,15)
        distort_image = image.rotate(random_angle, Image.BICUBIC)
        '''
        # random center crop
        random_scale = np.random.uniform(0.7,1)
        width, height = distort_image.size[0], distort_image.size[1]
        random_width, random_height = width*random_scale, height*random_scale
        width_offset, height_offset = (width-random_width)/2, (height-random_height)/2
        # (left, top, right, bottom)
        bounding_box = (width_offset, height_offset, width_offset+random_width, height_offset+random_height)
        distort_image = distort_image.crop(bounding_box)
        # resize to original size
        distort_image = distort_image.resize((width, height))
        '''
        # random flip
        random_flip = np.random.randint(0,3)
        if random_flip == 0:
            distort_image = distort_image.transpose(Image.FLIP_LEFT_RIGHT)
        elif random_flip == 1:
            distort_image = distort_image.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            pass

        # color jittering
        if is_colorful:
            distort_image = distort_color(distort_image)

        return distort_image

    # data augment
    distort_image = distort(image)
    # PIL to numpy.ndarray
    # plt.imshow(np.array(distort_image).astype(np.float32)/255.)
    # plt.show()
    if not is_colorful:
        distort_image = np.expand_dims(np.array(distort_image).astype(np.float32)/255., -1)
    else:
        distort_image = np.array(distort_image).astype(np.float32)/255.

    return distort_image

def pre_process(image):
    image = data_augment(image)
    return image

# test
if __name__=='__main__':
    mnist = input_data.read_data_sets("./mnist/", one_hot=True)
    datasource = get_data(mnist)
    data_gen = gen_batch_data(datasource, 10)
    for i in range(10):
        images, labels = next(data_gen)
        print(images.shape)
        print(labels.shape)
