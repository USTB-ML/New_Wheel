"""
初赛题的多标签解决方法，该方法已知最高能实现94.375%的准确率
请注意你们的TFrecords的读取，有问题的话可以换成其他的方法，目前还有点玄学
如果你们内存/显存不够了，可以减小batch_size试试，还不行就压缩图片大小到64*64
另外注意binary_crossentropy和categorical_crossentropy的区别
binary_crossentropy是multi-label任务从原理上唯一正确选择，Keras的官方acc无法正常显示，需要自定义
categorical_crossentropy只会找到三个输出神经元中第一个最大值(1)的地址，所以只会更新最前面的那个，这不正确
原有acc=0.84意味着真正的正确率约为0.84^3=0.59，而只有acc=98.08%才能接近94.35%
请大家多多思考，多多尝试，争取搞出点新想法而不是靠玄学来提升准确率
比如改一改网络的结构或者规模？（模型简单就不要太深，你用ResNet50基本也是这个正确率，梯度会消失的）
接下来，欢迎大家享受深度玄学调参之旅~~
welcome to the Beginning of Deep Dark Learning~~
"""
import keras
from keras.models import model_from_json
import tensorflow as tf
from keras import backend as K
from keras.applications.inception_v3 import inception_v3
from keras import Input, Model, layers
from keras.utils import np_utils
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.engine.topology import get_source_inputs
from keras.optimizers import Adam
test_path = "testTanh.tfrecords"    # TFrecords的名字改成你们自己的
train_path = "trainTanh.tfrecords"

batch_size = 32
num_classes = 3
epochs = 8
num_predictions = 3
input_shape = (128, 128, 3)
train_samples = 8000
val_samples = 8000


def imgs_input_fn(filenames, perform_shuffle=False, Repeats=epochs, Batchs=train_samples):
    def _parse_function(serialized):
        features = \
            {
                'raw_image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([num_classes], tf.int64)    # num_classes=0就空着[]
            }
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)
        # Get the image as raw bytes.
        image_shape = tf.stack([128, 128, 3])
        image_raw = parsed_example['raw_image']
        label = tf.cast(parsed_example['label'], tf.int32)
        # label = tf.one_hot(label, num_classes)
        # Decode the raw bytes so it becomes a tensor with type.
        image = tf.decode_raw(image_raw, tf.uint8)
        image = tf.cast(image, tf.float32) * 1./255 - 0.5
        image = tf.reshape(image, image_shape)
        image = tf.reverse(image, axis=[2])  # 'RGB'->'BGR'
        return image, label

    # 其实这个才是标准用法，然而莫名其妙的行不通，我也很绝望啊
    # imgs, labels = _parse_function(filenames)
    # # print(imgs, labels)
    # x_batch, y_batch = K.tf.train.batch([imgs, labels], batch_size=Batchs, capacity=100)
    # print(x_batch, y_batch)
    # return x_batch, y_batch

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(Batchs).repeat(Repeats)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    iterator = iterator.get_next()

    with tf.Session() as sess:  # 这里使用的是“大力出奇迹”，把所有数据都读入内存
        batch_features, batch_labels = sess.run(iterator)
        return batch_features, batch_labels
        # sess.run(tf.global_variables_initializer())  # 参数初始化，要用yield的话贼重要
        # while Run:
        #     batch_features, batch_labels = sess.run(iterator)
        #     yield {'input': batch_features}, {'output': batch_labels}



'''开始
以下是官网上ResNet的部分结构函数，直接COPY过来用的
'''
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor  #输入变量#
        kernel_size: defualt 3, the kernel size of middle conv layer at main path #卷积核的大小#
        filters: list of integers, the filterss of 3 conv layer at main path  #卷积核的数目#
        stage: integer, current stage label, used for generating layer names #当前阶段的标签#
        block: 'a','b'..., current block label, used for generating layer names #当前块的标签#
    # Returns
        Output tensor for the block.  #返回块的输出变量#
    """
    filters1, filters2, filters3 = filters  # 滤波器的名称#
    if K.image_data_format() == 'channels_last':  # 代表图像通道维的位置#
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)  # 卷积层，BN层，激活函数#

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    # conv_block is the block that has a conv layer at shortcut
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def DivResNet(include_top=True, weights='imagenet',
             input_tensor=None, pooling='max',
             classes=num_classes):  # 这里采用的权重是imagenet，可以更改，种类为1000#
    # Determine proper input shape
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)  # 对图片界面填充0，保证特征图的大小#
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)  # 定义卷积层#
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)  # 批标准化#
    x = Activation('relu')(x)  # 激活函数#
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 最大池化层#
    # stage2#
    x = conv_block(x, 3, [32, 32, 128], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='b')
    # stage3#
    x = conv_block(x, 3, [64, 64, 256], stage=3, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='b')
    # stage4#
    x = conv_block(x, 3, [128, 128, 512], stage=4, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='b')
    # stage5#
    x = conv_block(x, 3, [256, 256, 1024], stage=5, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=5, block='b')

    x = AveragePooling2D((4, 4), name='avr_pool')(x)  # 平均池化层#

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='sigmoid', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')
    return model

'''结束
以上是官网上ResNet50的部分结构函数，直接COPY过来用的
'''
def multi_pred(y_true, y_pred):
    acc = tf.floor(K.mean(K.equal(y_true, K.round(y_pred)), axis=-1))
    return acc

# The data, split between train and test sets:
# print(imgs_input_fn(train_path).shape)
x_train, y_train = imgs_input_fn(train_path)
x_test, y_test = imgs_input_fn(test_path, Batchs=val_samples)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# # Convert class vectors to binary class matrices.

# train_inp = Input(tensor=x_train)
model = DivResNet()
# model = keras.models.load_model('model3.h5', custom_objects={'multi_pred': multi_pred})

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['accuracy', multi_pred],
              # callbacks=[keras.callbacks.TensorBoard(log_dir='./log0', histogram_freq=1)],
              )
model.summary()
best_acc = []
for i in range(epochs):
    model.fit(x_train, y_train,
              batch_size=batch_size,
              shuffle=True,
              epochs=1,
              # validation_data=(x_test, y_test),
              verbose=1)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('epoch', i)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[2])
    best_acc.append((i, {'Multi-acc': scores[2]}))
    # Save the final model
    model_json = model.to_json()
    mdl_save_path = 'model'+str(i)+'.json'
    with open(mdl_save_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    mdl_save_path = 'model'+str(i)+'.h5'
    model.save(mdl_save_path)
    if i == 2:
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=1e-5),
                      metrics=['accuracy', multi_pred],
                      # callbacks=[keras.callbacks.TensorBoard(log_dir='./log1', histogram_freq=1)],
                      )
    scores = model.evaluate(x_train, y_train, verbose=0)
    print('epoch', i)
    print('Train loss:', scores[0])
    print('Train accuracy:', scores[2])

print(best_acc)
# 保存模型
model.save('myModel.h5')   # HDF5文件，pip install h5py
# 保存网络结构，载入网络结构
json_path = 'myModel.json'
json_string = model.to_json()
with open(json_path, "w") as json_file:
    json_file.write(json_string)
print(json_string)
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[2])

# 输出图
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
# 使用完模型之后，清空之前model占用的内存
K.clear_session()
tf.reset_default_graph()
