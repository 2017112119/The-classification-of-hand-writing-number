#coding:utf-8
#Author:JinGang Wang
#导入所需的库
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#数据加载
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#定义准确率计算方法
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)#产生随机变量进行初始化
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


#***********网络结构************---开始#


#定义  placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784])/255.  #28x28  归一化
ys = tf.placeholder(tf.float32,[None,10]) #输出
keep_prob = tf.placeholder(tf.float32)  #dropout
x_image = tf.reshape(xs,[-1,28,28,1])

##conv1 layer
W_conv1 = weight_variable([5,5,1,32])   #patch 5x5  in size 1 ,  out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)    #output size = 14 x 14 x 32

##conv2 layer
W_conv2 = weight_variable([5,5,32,64])   #patch 5x5  in size 32 ,  out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)    #output size = 7 x 7 x 64

## full connected 1 layer
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])#
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

## full connected 2 layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)


#***********网络结构************---结束#



#定义误差
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))#交叉熵
#优化
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#声明tf.train.Saver类用于保存模型
saver = tf.train.Saver()

#训练
for i in range(10000):
    batch_xs ,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i%50 == 0:
        acc = compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000])
        print(acc)
        if acc>0.99:
            print("训练了%d轮。"%i)
            break

saver.save(sess,"E:\学习\手写数字识别\model.ckpt")

# 0.133
# 0.752
# 0.862
# 0.886
# 0.908
# 0.917
# 0.925
# 0.92
# 0.925
# 0.936
# 0.942
# 0.941
# 0.95
# 0.955
# 0.953
# 0.956
# 0.958
# 0.96
# 0.965
# 0.962
# 0.964
# 0.963
# 0.968
# 0.967
# 0.971
# 0.967
# 0.967
# 0.972
# 0.97
# 0.975
# 0.974
# 0.977
# 0.973
# 0.976
# 0.974
# 0.98
# 0.977
# 0.975
# 0.976
# 0.98
# 0.976
# 0.982
# 0.979
# 0.984
# 0.981
# 0.98
# 0.984
# 0.981
# 0.981
# 0.981
# 0.984
# 0.985
# 0.982
# 0.978
# 0.985
# 0.985
# 0.984
# 0.984
# 0.984
# 0.983
# 0.984
# 0.981
# 0.985
# 0.985
# 0.983
# 0.989
# 0.981
# 0.986
# 0.985
# 0.981
# 0.984
# 0.986
# 0.983
# 0.987
# 0.986
# 0.987
# 0.988
# 0.984
# 0.987
# 0.984
# 0.988
# 0.987
# 0.988
# 0.986
# 0.988
# 0.989
# 0.987
# 0.991
# 训练了4350轮。
