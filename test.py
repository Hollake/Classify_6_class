#coding:utf-8
import time
import tensorflow as tf
import forward
import backward
import generateds
REGULARIZER = 0.0001
TEST_INTERVAL_SECS = 5
TEST_NUM = 4800
BATCH_SIZE = 50
TFR_save_dir = "./data/train"
'''

'''
def test():
    #复现之前的计算图
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, 128,128,3])
        y_ = tf.placeholder(tf.float32, [None, 6])
        y = forward.forward(x, REGULARIZER)
        # 实例化具有滑动平均的saver对象从而在会话被加载时模型
        # 中的所有参数被赋值为各自的滑动平均值，增强模型的稳定性，然后计算模型在
        # 测试集上的准确率。
        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #获得数据输入
        img_batch, label_batch = generateds.readTFReccord(TFR_save_dir)

# 在 with 结构中，加载指定路径下的 ckpt，若模型存在，则
# 加载出模型到当前对话，在测试数据集上进行准确率验证，并打印出当前轮数下
# 的准确率，若模型不存在，则打印出模型不存在的提示，从而 test()函数完成。
# 通过主函数 main()，加载指定路径下的测试数据集，并调用规定的 test 函数，
# 进行模型在测试集上的准确率验证。
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    coord = tf.train.Coordinator()#3
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)#4

                    xs, ys = sess.run([img_batch, label_batch])

                    accuracy_score = sess.run(accuracy, feed_dict={x: xs, y_: ys})

                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))

                    coord.request_stop()#6
                    coord.join(threads)#7

                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    test()#8

if __name__ == '__main__':
    main()
