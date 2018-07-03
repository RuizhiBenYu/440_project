import os
import tensorflow as tf
import pandas as pd
import train_test_split
import vec_to_array
import next_batch
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# data preparation
# pulling the table
train_x, train_rx, train_y, vali_x, vali_rx, vali_y, test_x, test_rx, test_y = next_batch.useful_data()


# define accuracy
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys:v_ys, keep_prob: 1})
    return result


# define weight and bias
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)


# define CONV and pooling
def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],  padding='SAME')


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])    #28 * 28
ys = tf.placeholder(tf.float32, [None, 5])
# define the dropout placeholder
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1,28,28,1])


## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# h_fc1_drop = tf.layers.dropout(h_fc1, rate=0.5)

## fc2 layer ##
W_fc2 = weight_variable([1024, 5])
b_fc2 = bias_variable([5])
# h_fc2 = tf.matmul(h_fc1_drop, W_fc2)
# h_fc_2_review = tf.matmul(h_fc2,)
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# loss = tf.reduce_mean(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
# if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#     init = tf.initialize_all_variables()
# else:
init = tf.global_variables_initializer()
sess.run(init)

# batch_xs, batch_x_review, batch_ys = next_batch.next_batch(100)

for i in range(1000):
    batch_xs, batch_ys = next_batch.next_batch(i, 100)
    # sess.run(train_step, feed_dict={xs: train_x, ys: train_y, keep_prob: 0.5})

    # cross_entropy_test = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    # _, loss_train = sess.run([train_step, loss], feed_dict={xs: train_x, ys: train_y, keep_prob: 0.5})
    _, loss_val = sess.run([train_step, cross_entropy], feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
       # print(i,",",compute_accuracy(test_x,test_y),",",loss)
       print("iteration", i, "accuracy_vali", compute_accuracy(test_x,test_y), "accuracy_train", compute_accuracy(train_x, train_y),"loss", loss_val)
       # print("loss", loss_train, "loss2", loss_val)
       # print()