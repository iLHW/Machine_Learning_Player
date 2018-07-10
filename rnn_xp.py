import tensorflow as tf

# 下载mnist数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/', one_hot=True)

# 一张图片是28*28,前馈神经网络是一次性把数据输入到网络，而RNN把它分成块
chunk_size = 28
chunk_n = 28

rnn_size = 256  # RNN大小

n_output_layer = 10  # 输出层

X = tf.placeholder('float', [None, chunk_n, chunk_size])  # 定义placeholder作为存放输入输出数据的地方
Y = tf.placeholder('float')


# 定义待训练的神经网络
def recurrent_neural_network(data):
    layer = {'w_': tf.Variable(tf.random_normal([rnn_size, n_output_layer])),  # 正态分布初始化权重 w 变量
             'b_': tf.Variable(tf.random_normal([n_output_layer]))}  # 正态分布初始化 bias 变量

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)  # 定义基本的LSTM神经元

    data = tf.transpose(data, [1, 0, 2])  # 处理数据及其格式
    data = tf.reshape(data, [-1, chunk_size])
    data = tf.split(data, chunk_n, 0)  # 新版tf这里第一个和第三个参数要对调!
    outputs, status = tf.nn.static_rnn(lstm_cell, data, dtype=tf.float32)  # 新版tf这里要用static_rnn,原版就是rnn

    ouput = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])  # 为避免冲突定义typo输出ouput

    return ouput


# 每次使用100条数据进行训练
batch_size = 100


# 使用数据训练神经网络
def train_neural_network(X, Y):
    predict = recurrent_neural_network(X)  # 预测初始化
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))  # 损失函数初始化
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # 优化器初始化

    epochs = 13  # 考虑实际效果取epochs为13
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())  # 运行会话
        epoch_loss = 0  # 初始化损失
        for epoch in range(epochs):
            for i in range(int(mnist.train.num_examples / batch_size)):
                x, y = mnist.train.next_batch(batch_size)  # 取批量
                x = x.reshape([batch_size, chunk_n, chunk_size])  # 格式变换
                _, c = session.run([optimizer, cost_func], feed_dict={X: x, Y: y})  # 喂数据并运行会话
                epoch_loss += c
            print(epoch, ' : ', epoch_loss)  # 输出进度与损失

        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率: ', accuracy.eval({X: mnist.test.images.reshape(-1, chunk_n, chunk_size), Y: mnist.test.labels}))


train_neural_network(X, Y)  # 开始训练循环神经网络
