import  tensorflow as tf
import  os
import random
from PIL import Image
import matplotlib.pyplot as plt
from scipy import misc
""""
这段代码主要是用TensorFlow来生成TFRecord文件

"""
#图片存储路径
picture_dir = "./validation"
#标签类别
classes = os.listdir(picture_dir)
#生成的TFRecord保存路径
TFR_save_dir = "./data/validation"
#每个TFRecord文件存放图片数量
pic_num = 2500
#图像通道数
channel = 3
#批次数
batchsize = 64
#保存图像路径
savepic = "./savepic/"
#样本总数
numtotal = 4800
#读取多少轮
epoch = 2

#获取图像路径及其对应的标签，并且打乱顺序.否则，
#同一个TFRecord文件中只会包含一个类别
#总共为6个标签，将其表示为one_hot 向量形式
def get_picture(picture_dir):
    pic_list = []
    label_list = []
    #先初始化为[0,0,0,0,0,0]
    label = [0]*6
    #进行遍历
    for index, name in enumerate(classes):
        class_path = picture_dir+'/'+name+'/'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            pic_list.append(img_path)
            #第一个类的标签的label置为[1,0,0,0,0,0],第三个为[0,0,1,0,0,0],以此类推
            label[index]=1
            label_list.append(label)
        label = [0]*6
    dict = list(zip(pic_list,label_list))
    random.shuffle(dict)
    return dict


def createTFRecord(picture_dir,TFR_save_dir):
    # TFRecord文件个数
    TFR_num = 2
    #文件计数器
    num = 0
    # 图像计数
    counter = 1
    # tfrecords格式文件名
    TFRcord_name = ("traindata-%.2d-%.2d.tfrecords" % (TFR_num,num))
    writer = tf.python_io.TFRecordWriter(TFR_save_dir+'/' + TFRcord_name)
    # 类别和路径
    dict = get_picture(picture_dir)
    for img_path, label in dict:
            counter = counter + 1
            print('The %4d pic has trans'%counter)
            #如果转换的图片数量大于文件定的最大数量，那么则将剩下的数据写入第二个文件
            if counter > pic_num:
                counter = 1
                num  = num + 1
                # tfrecords格式文件名
                TFRcord_name = ("traindata-%.2d-%.2d.tfrecords" % (TFR_num,num))
                writer = tf.python_io.TFRecordWriter(TFR_save_dir+'/' + TFRcord_name)
            img = Image.open(img_path, 'r')
            # img = misc.imresize(img, 0.5)
            #对照片进行剪裁，将256*256的剪裁为128*128大小的照片
            box=(64,64,192,192)
            img=img.crop(box)
            size = img.size
            img_raw = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                    'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]])),
                    'img_channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[channel]))
                }))
            writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()
    print("finish")

def readTFReccord(TFR_save_dir):
    data_files = tf.gfile.Glob(TFR_save_dir+'/' + 'traindata-*.tfrecords')
    # 如有多个TFRecord文件，则打乱顺序读取，有利于模型的训练
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([6], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'img_width': tf.FixedLenFeature([], tf.int64),
                                           'img_height': tf.FixedLenFeature([], tf.int64),
                                           'img_channel': tf.FixedLenFeature([], tf.int64),
                                       })
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    height = tf.cast(features['img_height'], tf.int32)
    width = tf.cast(features['img_width'], tf.int32)
    channel = tf.cast(features['img_channel'], tf.int32)
    label = tf.cast(features['label'], tf.float32)
    image = tf.reshape(image, [height, width, channel])
    #将照片大小设定为[128,128]
    image.set_shape([128,128,3])
    #若希望能够还原照片，则返回image,label
    # return image ,label


    # 如果要将数据喂给网络训练，则将这段代码注释去掉即可，返回值也改成相应的img_batch和label_batch
    image = tf.cast(image, tf.float32) * (1. / 255)#数据归一化
    img_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                    batch_size= batchsize,
                                                    num_threads=4,
                                                    capacity=1000,
                                                    min_after_dequeue=500)
    return img_batch, label_batch
def get_pic(TFR_save_dir):
#将数据从TFRecord文件中读取出来，可以显示，保存到文件夹
    img_batch,label_batch = readTFReccord(TFR_save_dir)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        #启动多线程
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(sess=sess,coord=coord)
        for i in range(4800):
            imgs,labs = sess.run([img_batch,label_batch])#在会话中取出imgs和labs
            imgs=Image.fromarray(imgs, 'RGB')#这里Image是之前提到的
            # plt.imshow(imgs)#显示图像
            # plt.show()
            imgs.save(savepic+"/"+str(i)+'_pic_''Label_'+str(labs)+'.jpg')#存下图片
            print('The %4d pic has recover'%i)
        coord.request_stop()
        coord.join(threads)
if __name__ == '__main__':
    createTFRecord(picture_dir,TFR_save_dir)
    # get_pic(TFR_save_dir)
    # readTFReccord(TFR_save_dir)
    # get_picture(picture_dir)