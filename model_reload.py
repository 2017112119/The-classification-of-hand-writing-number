#coding:utf-8
#Author:JinGang Wang

import tensorflow as tf
import cv2 as cv
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#load the data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#定义网络结构类
class net_structure:
    def __init__(self):
        #定义  placeholder for inputs to network
        self.xs = tf.placeholder(tf.float32,[None,784])/255.  #28x28  归一化
        self.ys = tf.placeholder(tf.float32,[None,10]) #输出
        self.keep_prob = tf.placeholder(tf.float32)  #dropout
        x_image = tf.reshape(self.xs,[-1,28,28,1])

        ##conv1 layer
        W_conv1 = self.weight_variable([5,5,1,32])   #patch 5x5  in size 1 ,  out size 32
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image,W_conv1)+b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)    #output size = 14 x 14 x 32

        ##conv2 layer
        W_conv2 = self.weight_variable([5,5,32,64])   #patch 5x5  in size 32 ,  out size 64
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1,W_conv2)+b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)    #output size = 7 x 7 x 64

        ## full connected 1 layer
        W_fc1 = self.weight_variable([7*7*64,1024])
        b_fc1 = self.bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])#
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1,self.keep_prob)

        ## full connected 2 layer
        W_fc2 = self.weight_variable([1024,10])
        b_fc2 = self.bias_variable([10])
        self.prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

        #模型加载
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess,'E:\学习\手写数字识别\model.ckpt')

    #创建权重矩阵
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape,stddev=0.1)#产生随机变量进行初始化
        return tf.Variable(initial)

    #创建偏置量向量
    def bias_variable(self,shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

    #卷积定义方法
    def conv2d(self,x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

    #池化定义方法
    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


#主函数进行测试  , 之所以要使用 main 函数  ，是因为这样在别的文件中 调用这个文件中的方法时，不会执行main函数中的语句
if __name__ == '__main__':

    #创建实例
    net = net_structure()

    #准确率计算方法定义
    def compute_accuracy(v_xs,v_ys):
        y_pre = net.sess.run(net.prediction,feed_dict={net.xs:v_xs,net.ys:v_ys,net.keep_prob:1})
        correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        result = net.sess.run(accuracy,feed_dict={net.xs:v_xs,net.ys:v_ys,net.keep_prob:1})
        return result

    #计算准确率
    acc = compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000])
    print(acc)

    #图像读取与预处理
    print(mnist.test.images[0])
    img1 = cv.imread('3.jpg')#读取
    img_resize = cv.resize(img1,(28,28))#改变大小
    img_gray = cv.cvtColor(img_resize,cv.COLOR_RGB2GRAY)#转换为灰度图
    cv.imshow("hh",img_gray)#显示
    cv.waitKey(0)
    cv.destroyAllWindows()

    #进行预测
    res = np.reshape(img_gray,784)
    y_pre = tf.argmax(net.prediction,1)#tf.argmax(矩阵，axis)  axis=1返回行中最大元素位置   axis=0 返回列中最大元素的位置
    predint = y_pre.eval(feed_dict={net.xs:[res],net.keep_prob:1},session=net.sess)

    print(predint)
    net.sess.close()
