# Classify_6_class\<br>
在train与validation下各有6个文件夹，train下面的每个文件夹下都有800张照片，validation每个文件夹下各有200张照片\<br>
generateds文件可以将图片转换为tfrecords文件，然后读出喂给神经网络或者再还原为照片本身，存入savepic文件下，请在运行前建好savepic文件夹以及data文件夹\<br>
data文件夹下有train与validation子目录。\<br>
在训练时请运行backward.py文件，输出loss，test输出准确率\<br>
实验结果不是很理想，应该是神经网络的各种参数不理想造成的\<br>
