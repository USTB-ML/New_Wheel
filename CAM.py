"""
此文件用于生成CAM图

原理：最后一层卷积含有图片高度抽象的信息，那么从该卷积层应该能生成能体现图像分类特征的图像

eg:
最后一层卷积：(7, 7, 512)
平均池化：(1, 1, 512)
输出层：(512, 14)

CAM：(7, 7, 512) * (512)   512个7*7图像与相关的权重（输出层的参数w）相乘然后叠加就是最终的图像


注意：受限于原理，模型必须要有全局平均池化，且池化后连接输出层
        由于使用OpenCV读写图片，默认状态下无法识别中文路径
"""
from keras.models import load_model, Model
import cv2
import numpy as np


# 将概率转化为标签
def one_hot_to_label(pre):
    """
    :param pre: 预测概率
    :return: 转化的标签
    """
    r = []
    one_num = 0
    for i in range(14):
        if pre[0][i] > 0.7:
            r.append('1')
            one_num += 1
        else:
            r.append('0')
    if one_num == 0:
        r.append('1')
    else:
        r.append('0')
    return r


def visualize_class_activation_map(model_path, img_path, output_path, model_row, model_col, all_label=False):
    """
    :param model_path: 选用模型的路径
    :param img_path: 要生成CAM图的图片路径
    :param output_path: CAM图输出路径
    :param model_row: 模型要求输入的图片宽
    :param model_col: 模型要求输入的图片高
    :param all_label: 是否输出对于该图像所有标签的CAM图像，默认为False， 仅输出预测为1的标签
    :return: None
    """

    # 载入模型
    model = load_model(model_path)

    # 载入图片
    original_img = cv2.imread(img_path, 1)
    width, height, _ = original_img.shape

    img = cv2.resize(original_img, (model_row, model_col), interpolation=cv2.INTER_NEAREST)
    img = np.array([img])

    # 获取最后一层的权重
    class_weights = model.layers[-1].get_weights()[0]

    # 获取最后一个卷积层的输出
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[-4].output)
    intermediate_output = intermediate_layer_model.predict(img)
    intermediate_output = intermediate_output[0, :, :, :]

    # 获取各标签输出概率
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[-1].output)
    predict_ = intermediate_layer_model.predict(img)

    label = one_hot_to_label(predict_)
    one_all = []

    if all_label:
        for i in range(len(label)):
            one_all.append(i)
    else:
        for i in range(len(label)):
            if label[i] == '1':
                one_all.append(i)

    # Create the class activation map.
    for show_label in one_all:

        # init CAM图
        cam = np.zeros(dtype=np.float32, shape=intermediate_output.shape[0:2])

        # 合成
        for i, w in enumerate(class_weights[:, show_label]):
            cam += w * intermediate_output[:, :, i]

        # 处理CAM图
        cam = abs(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (width, height))

        # 生成热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        img = heatmap * 0.4 + original_img

        # 保存
        cv2.imwrite(output_path+'_'+str(show_label)+'_label.png', img)


if __name__ == '__main__':
    visualize_class_activation_map('./out_model/ResNet34_0.23901808789380144.h5',
                                   './00030323_048_1.png',
                                   './00030323_048_enhance', 224, 224)
