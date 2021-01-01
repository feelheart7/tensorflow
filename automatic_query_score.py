# coding=utf-8
# 操作图片模块
from PIL import Image
# 范围随机模块
import random
# 操作系统模块
import os
# 矩阵计算numpy1.16与tensorflow1.12（深度学习框架）
import numpy as np
import tensorflow as tf

# TF_CPP_MIN_LOG_LEVEL默认值为 0 (显示所有logs)
# 设置为 1 隐藏 INFO logs, 2 额外隐藏WARNING logs
# 设置为3所有 ERROR logs也不显示
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 全局变量
IMAGE_HEIGHT = 22  # 验证码图片高度
IMAGE_WIDTH = 68  # 验证码图片宽度
MAX_CAPTCHA = 4  # 验证码的位数
CHAR_SET_LEN = 36  # 验证码的字符有多少种

# 训练数据集绝对存储路径
VERIFICATION_CODE_TRAINING_PATH = os.path.dirname(__file__) + '/verification_code_training_images/'
# placeholder 是 Tensorflow 中的占位符，暂时储存变量 X Y ,keep_prob是dropout层保留概率
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)

# 训练集所有图片list与图片大小
all_images = os.listdir(VERIFICATION_CODE_TRAINING_PATH)
all_images_size = len(all_images)


################################################################
### 通过tensorflow的CNN(卷积神经网络深度学习后识别验证码,识别率99.6%####
################################################################


# 获取验证码名字和图片（训练数据集）
def get_name_and_image():
    # 获取数据集下的所有图片的数组all_images
    # all_images = os.listdir(VERIFICATION_CODE_TRAINING_PATH)
    random_image = random.randint(0, all_images_size - 1)
    # print (all_images_size)
    base = os.path.basename(VERIFICATION_CODE_TRAINING_PATH + all_images[random_image])  # 有扩展名
    name = os.path.splitext(base)[0]  # 无扩展名
    image = Image.open(VERIFICATION_CODE_TRAINING_PATH + all_images[random_image])
    image = np.array(image)
    return name, image


# 验证码名字转变成向量: 不同位数的需要重写这个函数,函数里的数字为ASCII码
def name2vec(name):
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    for i, c in enumerate(name):
        if ord(c) < 58:
            idx = i * 36 + ord(c) - 48
            vector[idx] = 1
        else:
            idx = i * 36 + ord(c) - 87
            vector[idx] = 1
    return vector


# 向量转名字:注释部分是 最开始的向量 转 名字
# def vec2name(vec):
#     name = []
#     for i, c in enumerate(vec):
#         if c == 1.0:
#             name.append(i)
#     for i in range(0, 4):
#         if name[i] % 36 < 10:
#             name[i] = chr(name[i] % 36 + 48)
#         else:
#             name[i] = chr(name[i] % 36 + 87)
#     return "".join(name)


# 向量转名字: 训练是不用到这个函数，训练完成用这个函数得到最终结果
def vec2name(vec):
    name = []
    for i in vec:
        if i < 10:
            a = chr(i + 48)
            name.append(a)
        else:
            a = chr(i + 87)
            name.append(a)
    return "".join(name)


# 采样函数:默认一次采集64张验证码作为一次训练
# 需要注意通过get_name_and_image()函数获得的image是一个含布尔值的矩阵
# 在这里通过1*(image.flatten())函数转变成只含0和1的1行114*450列的矩阵
def get_next_batch(batch_size=64):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        name, image = get_name_and_image()
        batch_x[i, :] = 1 * (image.flatten())
        batch_y[i, :] = name2vec(name)
    return batch_x, batch_y


# 定义CNN(卷积神经网络):三个卷积层卷积神经网络结构
# 采用3个卷积层加1个全连接层的结构，在每个卷积层中都选用2*2的最大池化层和dropout层，卷积核尺寸选择5*5。，
# 我们的图片已经经过了3层池化层，也就是长宽都压缩了8倍(各自取整为3X9)
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    # 3个卷积层
    w_c1 = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([5, 5, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([5, 5, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # 1个全连接层
    w_d = tf.Variable(w_alpha * tf.random_normal([3 * 9 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


# 训练函数：选择sigmoid_cross_entropy_with_logits()交叉熵来比较loss
# 用adam优化器来优化
# keep_prob = 0.3，控制着过拟合
def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            batch_x, batch_y = get_next_batch(256)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.2})
            print(step, loss_)
            # 每100 step计算一次准确率
            if step % 1000 == 0:
                batch_x_test, batch_y_test = get_next_batch(1000)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                # 如果准确率大于99%,保存模型,完成训练
                if acc > 0.999:
                    saver.save(sess, "./crack_capcha.model", global_step=step)
                    break
            step += 1


train_crack_captcha_cnn()


# 训练完成后,注释train_crack_captcha_cnn()，取消下面的注释，开始预测，注意更改预测集目录
def crack_captcha():
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        n = 1
        while n <= 10:
            text, verification_code_training_images = get_name_and_image()
            verification_code_training_images = 1 * (verification_code_training_images.flatten())
            predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
            text_list = sess.run(predict, feed_dict={X: [verification_code_training_images], keep_prob: 1})
            vec = text_list[0].tolist()
            predict_text = vec2name(vec)
            print("正确: {}  预测: {}".format(text, predict_text))
            n += 1


crack_captcha()