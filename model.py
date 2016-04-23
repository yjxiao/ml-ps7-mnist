import tensorflow as tf


def _weight_var(shape, wd=0):
    """Wrap the variable definition process to include weight decay (l2 
    penalty).
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    var = tf.Variable(initial)
    # optional weigth decay (l2)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _bias_var(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
  

def inference(x, keep_prob):
    """Build the convnet model for inference

    Args:
        x_image: image placeholder
        keep_prob: dropout probability placeholder

    Returns:
        y_conv: probability output
    """
    # 1st convolutional layer
    W_conv1 = _weight_var([5, 5, 1, 32])
    b_conv1 = _bias_var([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 2nd convolutional layer
    W_conv2 = _weight_var([5, 5, 32, 64])
    b_conv2 = _bias_var([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # fully connected layer
    W_fc1 = _weight_var([7*7*64, 512])
    b_fc1 = _bias_var([512])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # add dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # fully connected + softmax layer
    W_fc2 = _weight_var([512, 10])
    b_fc2 = _bias_var([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv


def loss(y_conv, y):
    """Calculates the cross entropy loss"""
    cross_entropy = - tf.reduce_mean(y * tf.log(y_conv), name='xentropy')
    tf.add_to_collection('losses', cross_entropy)
    # total loss including weight decay
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return cross_entropy, loss


def training(loss, lr):
    """Define the training operation using basic SGD"""
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(y_conv, y):
    """Evaluation operation to calculate accuracy"""
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy    

