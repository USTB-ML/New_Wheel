"""
此文件用来从csv文件中读取标签并从相关图片目录中读取图片,
允许设定最大读取量，如果设定最大读取量则需要传入总类数，以保证每一个类别都有近似相同数量的样本

相关函数有两个
read_image():从对应目录中按照相关要求读取图片并以np.array的形式返回图片数据
load_data_use_csv():从csv文件中读取标签、从read_image()请求图片数据并返回对应图片和标签

This file is used to read labels from CSV files and pictures from related picture directories
Maximum reads are allowed, and if maximum reads are set, the total number of classes will need to be passed in to
ensure that each class has approximately the same number of samples.

There are two correlation functions.
Read_image(): Read the picture from the corresponding directory according to the relevant requirements and
              return the picture data in the form of np.array

Load_data_use_csv(): Read labels from CSV files, request image data from read_image() and
                     return corresponding pictures and labels



History:

++++++++++++++++++++++++++++create+++++++++++++++++++++++++
Author: Koolo233 <Koolo233@163.com>
Created: 2018-11-24
++++++++++++++++++++++++++++update+++++++++++++++++++++++++
Author:

"""

import numpy as np
import os
import pandas as pd
import cv2


def read_image(img_name, IM_WIDTH, IM_HEIGHT, flag_colorful=True):

    """
    :param img_name: 图片名
                     image name
    :param IM_WIDTH: 图片的宽度(像素)
                      the width of image
    :param IM_HEIGHT: 图片的高度(像素)
                      the height of image
    :param flag_colorful: 是否为3通道图片，True为是，False为单通道， 默认为True
                           whether 3 channels
                           True: 3 channels
                           False: 1 channels
    :return: 以np.array的形式返回图片数据
              return the data of image in the form of np.array
    """

    if flag_colorful:
        im = cv2.imread(img_name, 1)
    else:
        im = cv2.imread(img_name, 0)

    im = cv2.resize(im, (IM_WIDTH, IM_HEIGHT), interpolation=cv2.INTER_NEAREST)
    data = np.array(im)

    return data


def load_data_use_csv(image_path, label_path, csv_col_name, csv_col_value, IM_WIDTH, IM_HEIGHT, max_load=None,
                      flag_colorful=True, n_class=None):

    """
    :param image_path: 图片所在文件夹路径
                        the path of images
    :param label_path: csv文件所在路径
                        the path of labels
    :param csv_col_name: csv文件 图片名列名
                        the column name of images in CSV file
    :param csv_col_value: csv文件 标签名列名
                        the column name of labels in CSV file
    :param IM_WIDTH: 图片设定宽度(像素)
                        resize images' width
    :param IM_HEIGHT: 图片设定高度(像素)
                        resize images' height
    :param max_load: 最大图片加载量，默认为加载全部
                        the max number of loading
                        default:all
    :param flag_colorful: 是否为3通道，True为是3通道，False为1通道，默认为True
                        whether 3 channels
                        True: 3 channels
                        else: 1 channels
                        default: True
    :param n_class: 总分类数，如果指定最大图片加载量，则需要传入此值，每一种类别的图片返回的数量为 int(max_load/n_class)
                    the number of classes
                    if setting the max number of loading, this param is necessary
                    the number of each labels will be int(max_load/n_class)
    :return: 以np.array的形式返回数据和标签 数据在前、标签在后
            return the data in the form of np.array, (images. labels)
    """

    print('Begin to load data')

    print('load csv file')
    result_frame = pd.read_csv(label_path)
    result_dist = dict(zip(result_frame[csv_col_name].values, result_frame[csv_col_value].values))
    print('csv file loads successfully')

    images = []
    label = []
    image_load_every_label = []
    img_load_num = 0

    # 是否设定最大读入量
    if max_load and n_class:
        for i in range(n_class):
            image_load_every_label.append(0)
        max_load_everylabel = int(max_load/n_class)

    print('Begin to load images')
    files = os.listdir(image_path)
    const_image_format = [".jpg", ".jpeg", ".bmp", ".png"]
    for fn in files:
        if os.path.splitext(fn)[1] in const_image_format:

            if max_load and n_class:
                if image_load_every_label[int(result_dist[fn[:-4]])] <= max_load_everylabel:

                    fd = os.path.join(image_path, fn)
                    images.append(read_image(fd, IM_WIDTH, IM_HEIGHT, flag_colorful))
                    label.append(result_dist[fn[:-4]])
                    img_load_num += 1
                    image_load_every_label[int(result_dist[fn[:-4]])] += 1

                    if img_load_num % 1000 == 0:
                        print(img_load_num)
            else:
                fd = os.path.join(image_path, fn)
                images.append(read_image(fd, IM_WIDTH, IM_HEIGHT, flag_colorful))
                label.append(result_dist[fn[:-4]])
                img_load_num += 1

                if img_load_num % 1000 == 0:
                    print(img_load_num)

        if max_load and n_class and img_load_num == max_load:
            break

    print('load success!')
    X = np.array(images)
    y = np.array(label)
    return X, y
