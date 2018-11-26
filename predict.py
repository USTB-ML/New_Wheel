from keras.preprocessing import image
import numpy as np
import os
from keras.models import load_model
path_pic = r'train_dir'
path_csv = r'csv_train.csv'
path_pic_test = r'test'
path_csv_test = r'csv_test.csv'
cwd = os.getcwd()
name_list = []


def read_data(is_train=False):
    # 当时为了调试搞了个可以从训练集读图片的参数，你们就当is_train没用就好
    path_list = []
    i = 0
    for name in os.listdir(path_pic_test):
        if is_train:
            image_path = os.path.join(path_pic, name)
        else:   # 从测试集中读取图片名
            image_path = os.path.join(path_pic_test, name)
        if os.path.isfile(image_path):
            path_list.append(image_path)
            item = name.strip().split(".")[0]   # 把.jpg删了
            name_list.append(item)
            if i % 1000 == 0:   # 每隔1000输出一下，看一看有没有幺蛾子
                print(i, path_list[i])
                # image.show()
            i = i + 1
        else:
            print("not:", image_path)
    return path_list


def preprocess_input(pic):
    # 标准化处理，你训练前对图片干过什么就再来一遍
    pic = np.expand_dims(pic, axis=0)   # 拓展一维，相当于告知batch=1
    pic = pic*1./255-0.5
    return [pic]


# 本实例为多标签任务(multi-labels)，所以用的是sigmoid
# softmax输出的答案请寻找最大值所在的位置（的字典），对应的即标签
def one_hot_to_label(pre):
    # 将预测的概率pre转化为明确的态度！
    r = []
    for i in range(3):  # 由于pre相当于batch=1的输出，所以要拆掉第一个括号
        if pre[0][i] > 0.5:
            r.append('1')
        else:
            r.append('0')
    return r


i = 0
y_pre = []
# 加载测试集图片目录
img_path = read_data(is_train=False)
model = load_model('model5.h5')
# 你所需要载入的模型
with open("myTry3.csv", "w") as f:
    # 第一个writelines添加标题，请按要求瞎改
    f.writelines("name,square,circle,triangle\n")
    for path in img_path:
        img = image.load_img(path, target_size=(128, 128))  # 注意大小
        if i % 500 == 0:    # 测试用，输出个图片人工观察分类
            img.show()
        x = image.img_to_array(img)
        x = preprocess_input(x)
        preds = model.predict(x)            # 利用Keras的函数来预测
        label = one_hot_to_label(preds)     # 将预测概率转换为标签
        y_pre.append(label)
        print(i, name_list[i], label, preds)
        # 请注意，下面的writlines只能输出字符串，是数字请强制转换，类似str(label)这样的
        f.writelines([name_list[i], ',', label[0], ',', label[1], ',', label[2], '\n'])
        if i % 500 == 0:    # 测试用，输出预测信息让你来检查图片是否分类正确
            print(path, label, preds)
        i = i + 1
