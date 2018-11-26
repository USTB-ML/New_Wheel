from PIL import Image
import os
import cv2  # 据说cv也行？没试过，好像有点问题
import tensorflow as tf
path_pic = r'train_save_pic'
path_csv = r'train.csv'
path_pic_test = r'test_save_pic'    # 其实这个叫验证集好一些
path_csv_test = r'testALL.csv'  # 当时初学就写错名字了
cwd = os.getcwd()
train_list = []     # 这里的思路是先读csv再找图片
test_list = []      # 反过来其实也行，自己看着办
num_classes = 3     # 类别数，注意修改

with open(path_csv) as csv:     # 读行，其实这个可以放到后面合并，懒得改了……
    i = 0
    alllines = csv.readlines()
    for line in alllines[1:]:
        train_list.append(line)
        i += 1

with open(path_csv_test) as csv:
    i = 0
    alllines = csv.readlines()
    for line in alllines[1:]:
        test_list.append(line)
        i += 1


'''
划重点！这里是提前将label转化为你所需的编码的地方
请根据自己的需求自由更改，如果你不需要one_hot或者
希望读入数据后再one_hot，请不要调用此函数
另外，这里演示的是多标签任务(multi-labels)的编码
'''
def int_2_one_hot(labels):
    r = []
    for i in range(num_classes):
        if labels[i] == '1':
            r.append(1)
        else:
            r.append(0)
    return r


def image_2_tfrecords(listname, tf_record_path, pic_path=path_pic):
    # listname是你的 [图片路径,标签们] 的list
    # tf_recored_path是你的TFrecords的名字（加上路径也行，没有则默认当前目录）
    # 第三个是图片文件夹的路径，只有文件夹名默认当前目录
    tf_write = tf.python_io.TFRecordWriter(tf_record_path)
    for i in range(len(listname)):
        item = listname[i]
        item = item.strip('\n')
        items = item.split(',')
        image_name = items[0] + ".jpg"
        image_path = os.path.join(pic_path, image_name)
        if os.path.isfile(image_path):
            image = Image.open(image_path)
            image = image.resize((128, 128))    # 注意修改成适合你的尺寸
            image = image.tobytes()
            labels = int_2_one_hot(items[1:])   # 第一行是标题不要。注意选择是否需要
            print(i, image_path, labels) 
            features = {}
            features['raw_image'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
            features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=labels))   # 如果传递数字，请加括号
            tf_features = tf.train.Features(feature=features)
            example = tf.train.Example(features=tf_features)
            tf_serialized = example.SerializeToString()
            tf_write.write(tf_serialized)
        else:
            print("not:", image_path)
    tf_write.close()


image_2_tfrecords(train_list, r"train.tfrecords")
image_2_tfrecords(test_list, r"test.tfrecords", pic_path=path_pic_test)
