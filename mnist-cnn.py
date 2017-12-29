from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))


def init_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def init_relu(x, bias):
    return tf.nn.relu(x + bias)


def init_conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def init_max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def inference(X, w1, w2, w3, w4, b1, b2, b3, b4):
    X = tf.reshape(X, [-1, 28, 28, 1])
    conv1 = init_conv2d(X, w1)
    relu1 = init_relu(conv1, b1)
    pool1 = init_max_pool(relu1)

    conv2 = init_conv2d(pool1, w2)
    relu2 = init_relu(conv2, b2)
    pool1 = init_max_pool(relu2)

    reshaped = tf.reshape(pool1, [-1, 7 * 7 * 64])

    fc3 = tf.nn.relu(tf.matmul(reshaped, w3) + b3)
    fc4 = tf.matmul(fc3, w4) + b4

    return fc4


batch_size = 128
test_size = 256

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

w1 = init_weight([5, 5, 1, 32])
w2 = init_weight([5, 5, 32, 64])
w3 = init_weight([7 * 7 * 64, 120])
w4 = init_weight([120, 10])
b1 = init_bias([32])
b2 = init_bias([64])
b3 = init_bias([120])
b4 = init_bias([10])

logit = inference(X, w1, w2, w3, w4, b1, b2, b3, b4)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=tf.argmax(Y, 1))
loss = tf.reduce_mean(cross_entropy)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(batch_size)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={X: batch[0], Y: batch[1]})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(train_op, feed_dict={X: batch[0], Y: batch[1]})
    print("test accuracy %g" % accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    saver.save(sess, 'model/')
