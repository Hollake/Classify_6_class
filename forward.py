import tensorflow as tf 
import numpy as np 

'''
由于输入矩阵过大，vgg16模型的输入为256*256的矩阵，这里输入72*398，所以我用了三个卷积层
三个全连接层，但是效果如何不知道
'''
# 生成权重函数，如果有正则化系数不为空，将正则化损失添加到losses中
def weight_variable(shape, regularizer):
	w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w
#偏置函数
def bias_variable(shape):  
	b = tf.Variable(tf.zeros(shape))  
	return b
#最大池化，周边自动填充
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],strides=[1, 4, 4, 1], padding='SAME')
#卷积运算
def conv2d(x, W):    

	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#卷积神经网络前向传播
'''
'''
def forward(x, regularizer):
#将输入序列reshape
	x_vt = tf.reshape(x, [-1, 128, 128, 3])
#第一卷积池化层256
	W_conv1 = weight_variable([5, 5, 3, 32],regularizer )
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_vt, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)   
#第二卷积池化层64
	w_conv2 = weight_variable([5,5,32,64],regularizer )
	b_conv2  = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

#第一全连接层
	W_fc1 = weight_variable([8*8*64, 128],regularizer )
	b_fc1 = bias_variable([128])   
	h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
	full_1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# #第二全连接层
	W_fc2 = weight_variable([128,6],regularizer)
	b_fc2 = bias_variable([6])   
	y_conv = tf.matmul(full_1, W_fc2) + b_fc2


#返回预测结果
	return y_conv