import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

IMAGE_PIXELS = 784
NUM_CLASSES = 10

def inference(image,hidden_units1,hidden_units2):
    with tf.name_scope("hidden1"):
        W = tf.Variable(tf.truncated_normal([IMAGE_PIXELS,hidden_units1],stddev=1.0/28),name="weight")
        b = tf.Variable(tf.zeros([hidden_units1]),name="bias")
        hidden1 = tf.nn.relu(tf.matmul(image,W)+b)

    with tf.name_scope("hidden2"):
        W = tf.Variable(
            tf.truncated_normal([hidden_units1, hidden_units2], stddev=1.0/math.sqrt(float(hidden_units1))), name="weight"
        )
        b = tf.Variable(tf.zeros([hidden_units2]), name="bias")
        hidden2 = tf.nn.relu(tf.matmul(hidden1,W)+b)

    with tf.name_scope("output_units"):
        W = tf.Variable(
            tf.truncated_normal([hidden_units2,NUM_CLASSES],stddev=1.0/math.sqrt(float(hidden_units2))),name="weight"
        )
        b = tf.Variable(tf.zeros([NUM_CLASSES]),name="bias")
        y_e = tf.nn.softmax(tf.matmul(hidden2,W)+b)
    return y_e

def loss(y_e,y):
    return -tf.reduce_sum(y*tf.log(y_e),name="crossEntropy")

def train(loss,learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    globla_step = tf.Variable(0,name="globla_step",trainable=False)
    return optimizer.minimize(loss,global_step=globla_step)

def correct(y_e,y):
    return tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_e,1)),dtype=tf.float32)
    )


if __name__ == "__main__":
    batch_size = 100
    hidden_units1 = 100
    hidden_units2 = 100
    learning_rate = 0.005

    image_batch = tf.placeholder(dtype=tf.float32,shape=[None,784])
    label_batch = tf.placeholder(dtype=tf.float32,shape=[None,10])

    y_e = inference(image_batch,hidden_units1,hidden_units2)
    loss = loss(y_e,label_batch)
    train_step = train(loss,learning_rate)
    correct = correct(y_e,label_batch)

    sess = tf.Session()
    init = tf.global_variables_initializer()

    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={image_batch: batch_xs, label_batch: batch_ys})
        if (i%100==0 and i!=0):
            print(sess.run(correct,feed_dict={image_batch:mnist.test.images,label_batch: mnist.test.labels}))





