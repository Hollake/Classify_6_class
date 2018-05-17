import tensorflow as tf
import forward
import os
import generateds

TFR_save_dir = "./data/train"
BATCH_SIZE = 50
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 20000
train_num_examples = 4800
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./model/"
MODEL_NAME="train_model"

#卷积神经网络反向传播函数
def backward():
	# 给输入x和真实标签y_占位
	x = tf.placeholder(tf.float32, [None,128,128,3])
	y_ = tf.placeholder(tf.float32, [None, 6 ])
	# 由前向传播得到预测值
	y = forward.forward(x, REGULARIZER)
	global_step = tf.Variable(0, trainable=False) 
	#定义均方误差
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
	cem = tf.reduce_mean(ce)
	# 定义损失，loss等于均方误差加上l2正则化误差
	loss = cem + tf.add_n(tf.get_collection('losses'))
	#定义指数衰减学习率学习率
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		train_num_examples / BATCH_SIZE, 
		LEARNING_RATE_DECAY,
		staircase=True)
	#定义反向传播算法
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	#定义滑动平均
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	ema_op = ema.apply(tf.trainable_variables())
	with tf.control_dependencies([train_step, ema_op]):
		train_op = tf.no_op(name='train')
	#实例化saver
	saver = tf.train.Saver()
	#从generateds.py文件得到批量数据
	img_batch, label_batch = generateds.readTFReccord(TFR_save_dir)
	#开始图计算
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		#实例化ckpt，在路径MODEL_SAVE_PATH下查找模型，如果有则加载模型
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		#开启线程协调器
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		#进行训练
		for i in range(STEPS):
			xs, ys = sess.run([img_batch, label_batch])
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
			if i % 1000 == 0:
				print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
				#每训练1000次保存模型
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
		#关闭线程协调器
		coord.request_stop()
		coord.join(threads)


def main():
	backward()

if __name__ == '__main__':
	main()


