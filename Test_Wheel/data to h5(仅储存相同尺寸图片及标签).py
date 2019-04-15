import os
import h5py
import numpy as np
import cv2

dir_list = [r'D:/dataset/hand/0', r'D:/dataset/hand/1', r'D:/dataset/hand/2',
            r'D:/dataset/hand/3', r'D:/dataset/hand/4', r'D:/dataset/hand/5']


def image_2_h5():

    Y = []
    X = []

    for dir in dir_list:
        dirs = os.listdir(dir)
        print(len(dirs))

        num = 0
        print('begin to process data in ' + dir)
        for file in dirs:
            label = eval(dir[-1])

            Y.append(label)

            im = cv2.imread(dir+'/'+file)
            im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
            mat = np.asarray(im)
            X.append(mat)

            num += 1
            if num % 500 == 0:
                print('have process ' + str(num) + ' data point')
                print(label)

    file = h5py.File(r'D:/dataset/hand/data.h5', 'w')
    file.create_dataset('X', data=np.array(X))
    file.create_dataset('Y', data=np.array(Y))
    file.close()

    # test
    data = h5py.File(r'D:/dataset/hand/data.h5', 'r')
    X_data = data['X']
    print(X_data.shape)
    Y_data = data['Y']
    print(Y_data[2])
    cv2.imshow('test', X_data[2])
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    image_2_h5()
