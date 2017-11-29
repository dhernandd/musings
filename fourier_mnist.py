# Fourier MNIST
from __future__ import print_function

import numpy as np

import tensorflow as tf

NUM_CLASSES = 10
BATCH_SIZE = 50


def create_variables(name, shape, initializer, device='CPU'):
    """
    TODO: Add GPU capabilities
    """
    if device == 'CPU':
        with tf.device(':/cpu:0'):
            dtype = tf.float32 
            var = tf.get_variable(name, shape, 
                                  initializer=initializer, dtype=dtype)
    return var


def fourier_process_batch(batch):
    """
    """
    fftR, fftI = np.zeros(batch.shape), np.zeros(batch.shape)
    for j in range(len(batch)):
        img = np.concatenate([batch[j], batch[j]], 1)
        fft = np.fft.rfft2(img)[:,1:]
        R, I = np.real(fft), np.imag(fft)
        fftR[j], fftI[j] = R/np.amax(R), I/np.amax(I)
    return fftR, fftI
             

def build_model(images):
    """
    """
    with tf.variable_scope('conv1') as scope:
        kernel = create_variables('weights', [5, 5, 3, 32], 
                        tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = create_variables('biases', [32], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        
    with tf.variable_scope('conv2') as scope:
        kernel = create_variables('weights', [5, 5, 32, 32], 
                        tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = create_variables('biases', [32], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    _, d1, d2, d3 = pool2.get_shape().as_list()
    
    with tf.variable_scope('full1') as scope:
        dim = d1*d2*d3
        reshape = tf.reshape(pool2, [tf.shape(pool2)[0], dim])
        weights = create_variables('weights', shape=[dim, 384],
                    initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = create_variables('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        
    with tf.variable_scope('softmax_linear') as scope:
        weights = create_variables('weights', [384, NUM_CLASSES],
                        tf.truncated_normal_initializer(stddev=0.02, dtype=tf.float32))
        biases = create_variables('biases', [NUM_CLASSES],
                                    tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local3, weights), biases, name=scope.name)

    return softmax_linear


def train_batch(inputs, outputs):
    preds = build_model(inputs)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=outputs, logits=preds))
    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)    
            
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(outputs, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return train_step, accuracy


def activation_summary():
    """
    Create activation summaries
    
    TODO:
    """
    pass


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 28, 28, 3], 'images')
    y = tf.placeholder(tf.int32, [None, 10], 'labels')
    train_step, accuracy = train_batch(x, y)
    
    ep = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            batch = list(mnist.train.next_batch(BATCH_SIZE))
            batch[0] = np.reshape(batch[0], [BATCH_SIZE, 28, 28])
            fftR, fftI = fourier_process_batch(batch[0])
            batch[0] = np.moveaxis(np.array([batch[0], fftR, fftI]), 0, -1)
#             batch[0] = np.moveaxis(np.array([batch[0], batch[0], batch[0]]), 0, -1)
            train_step.run(feed_dict={x : batch[0], y : batch[1]})
            if ep % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x : batch[0], y : batch[1]})
                print('step %d, training accuracy %.2f' % (ep, train_accuracy))
            if ep == 100:
                break
            ep += 1
            
        test = list(mnist.test.next_batch(5000))
        test[0] = np.reshape(test[0], [5000, 28, 28])
        fftR, fftI = fourier_process_batch(test[0])
        test[0] = np.moveaxis(np.array([test[0], fftR, fftI]), 0, -1)
        tl = np.mean(accuracy.eval(feed_dict={x : test[0], y : test[1]}))
        print('step %d, test accuracy %.4f' % (ep, tl))
    
    

    
