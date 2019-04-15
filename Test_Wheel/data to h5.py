import os
import h5py
import numpy as np
import cv2
import pandas as pd
import zipfile
import tarfile
import sys

dir_list = [r'D:/dataset/hand/0', r'D:/dataset/hand/1', r'D:/dataset/hand/2',
            r'D:/dataset/hand/3', r'D:/dataset/hand/4', r'D:/dataset/hand/5']


def cv_imread(img_path):
  file_path_gbk=img_path.encode('gbk')
  img_mat=cv2.imread(file_path_gbk.decode())
  return img_mat


def img_lab_2_h5_file(img_path, label_path, img_height, img_width, save_path, whether_check=True, check_point=500,
                      whether_test=True, image_column_name_in_h5='X', labels_column_name_in_h5='Y'):

    Y = []
    X = []
    image_name_list = []

    # load images and labels
    img_dirs = os.listdir(img_path)

    print('begin to process data in ' + img_path)
    loop = 0
    for file in img_dirs:
        filename, extension = os.path.splitext(file)
        image_name_list.append(filename)

        im = cv_imread(os.path.join(img_path, file))
        im = cv2.resize(im, (img_height, img_width), interpolation=cv2.INTER_AREA)
        mat = np.asarray(im)
        X.append(mat)
        loop += 1
        if whether_check:
            if loop % check_point == 0:
                print('have process ' + str(loop) + ' data point')

    data = pd.read_csv(label_path)
    for file in image_name_list:
        Y.append(data[file])

    # for dir in dir_list:
    #     dirs = os.listdir(dir)
    #
    #     num = 0
    #     print('begin to process data in ' + dir)
    #     for file in dirs:
    #         label = eval(dir[-1])
    #
    #         Y.append(label)
    #
    #         im = cv2.imread(dir+'/'+file)
    #         im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
    #         mat = np.asarray(im)
    #         X.append(mat)
    #
    #         num += 1
    #         if num % 500 == 0:
    #             print('have process ' + str(num) + ' data point')
    #             print(label)

    file = h5py.File(save_path, 'w')
    file.create_dataset(image_column_name_in_h5, data=np.array(X))
    file.create_dataset(labels_column_name_in_h5, data=np.array(Y))
    file.close()

    # test
    if whether_test:
        data = h5py.File(save_path, 'r')
        X_data = data[image_column_name_in_h5]
        print("The shape of images data is " + X_data.shape)
        Y_data = data[labels_column_name_in_h5]
        print("The shape of labels data is " + Y_data.shape)
        data.close()
        print("Check data")
        cv2.imshow('test', X_data[2])
        print(Y_data[2])
        if cv2.waitKey(0) & 0xFF:
            cv2.destroyAllWindows()


def img_2_h5_file(img_path, img_height, img_width, save_path, whether_check=True, check_point=500,
                  whether_test=True, image_column_name_in_h5='X'):

    X = []

    # load images and labels
    img_dirs = os.listdir(img_path)

    print('begin to process data in ' + img_path)
    loop = 0
    for file in img_dirs:

        im = cv_imread(os.path.join(img_path, file))
        im = cv2.resize(im, (img_height, img_width), interpolation=cv2.INTER_AREA)
        mat = np.asarray(im)
        X.append(mat)
        loop += 1
        if whether_check:
            if loop % check_point == 0:
                print('have process ' + str(loop) + ' data point')

    file = h5py.File(save_path, 'w')
    file.create_dataset(image_column_name_in_h5, data=np.array(X))
    file.close()

    # test
    if whether_test:
        data = h5py.File(save_path, 'r')
        X_data = data[image_column_name_in_h5]
        print("The shape of images data is " + X_data.shape)
        data.close()
        print("Check data")
        cv2.imshow('test', X_data[2])
        if cv2.waitKey(0) & 0xFF:
            cv2.destroyAllWindows()


def tar(fname):
    t = tarfile.open('D:\dataset\字符\\test' + ".tar.gz", "w:gz")
    for root, dir, files in os.walk(fname):
        print(root, dir, files)
        for file in files:
            fullpath = os.path.join(root, file)
            t.add(fullpath)
    t.close()


def img_2_h5_zip(zip_path, save_path, whether_check=True, check_point=500,
                 whether_test=True, image_column_name_in_h5='X', whether_resize=False, img_height=None, img_width=None):

    size_list = {}
    X_list = {}
    key_list = []

    X = []
    loop = 0

    if os.path.splitext(zip_path)[1] == '.gz':
        azip = tarfile.open(zip_path)
        file_names = azip.getnames()

        print('begin to process data')

        for n in file_names:
            img = azip.extractfile(n)
            if img:
                img = img.read()
                if sys.getsizeof(img) > 266:
                    # print(sys.getsizeof(img))
                    na = np.frombuffer(img, dtype=np.uint8)
                    im = cv2.imdecode(na, cv2.IMREAD_COLOR)
                    if whether_resize:
                        try:
                            im = cv2.resize(im, (img_height, img_width), interpolation=cv2.INTER_AREA)
                        except ValueError as e:
                            print()
                    mat = np.asarray(im)
                    X.append(mat)
                    loop += 1
                    if whether_check:
                        if loop % check_point == 0:
                            print('have process ' + str(loop) + ' data point')
        print("process over")

    else:
        azip = zipfile.ZipFile(zip_path)
        print(azip.namelist())

    # 按照 size 分组
    X = np.array(X)
    for img in X:
        if str(img.shape) not in size_list:
            print(str(img.shape))
            if str(img.shape) == "()":
                print(img)
            else:
                key_list.append(str(img.shape))
                size_list[str(img.shape)] = 0
                X_list[str(img.shape)] = []
                X_list[str(img.shape)].append(img)
        else:
            size_list[str(img.shape)] += 1
            X_list[str(img.shape)].append(img)

    file = h5py.File(save_path, 'w')
    for size in size_list:
        file.create_dataset(size, data=np.array(X_list[size]))
    file.close()

    # test
    if whether_test:
        data = h5py.File(save_path, 'r')
        X_data = data[key_list[0]]
        print("The shape of images data is " + str(X_data.shape))

        print("Check data")
        print(X_data)
        cv2.imshow('test', X_data[0])
        if cv2.waitKey(0) & 0xFF:
            cv2.destroyAllWindows()
        data.close()


if __name__ == '__main__':
    img_2_h5_zip(zip_path="D:\dataset\字符\image_contest_level_1_validate.tar.gz",
                 save_path="D:\dataset\字符\iimage_contest_level_1_validate.h5")
