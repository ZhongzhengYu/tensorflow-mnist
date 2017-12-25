from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def train(mnist):
    x_input = tf.placeholder(tf.float32, [None, 784])
    y_input = tf.placeholder(tf.float32, [None, 10])

    Weights1 = tf.Variable(tf.random_normal([784, 500]))
    biases1 = tf.Variable(tf.random_normal([500]))
    Weights2 = tf.Variable(tf.random_normal([500, 10]))
    biases2 = tf.Variable(tf.random_normal([10]))
    y_ = tf.nn.relu(tf.matmul(x_input, Weights1) + biases1)
    y_pre = tf.matmul(y_, Weights2) + biases2

    regularizer = tf.contrib.layers.l2_regularizer(0.001)
    regularization = regularizer(Weights1) + regularizer(Weights2)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pre, labels=tf.argmax(y_input, 1))) + regularization

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.9)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        validate_feed = {x_input: mnist.validation.images, y_input: mnist.validation.labels}
        test_feed = {x_input: mnist.test.images, y_input: mnist.test.labels}
        for i in range(10000):
            if i % 1000 == 0:
                print(s.run(accuracy, feed_dict=validate_feed))
            batch_xs, batch_ys = mnist.train.next_batch(100)
            s.run(optimizer, feed_dict={x_input: batch_xs, y_input: batch_ys})
        test_acc = s.run(accuracy, feed_dict=test_feed)
        print("After %d training step, test accuracy using average model is %g " % (10000, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
