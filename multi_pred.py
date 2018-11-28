'''	
经检验，model.fit的实时准确率acc并不准确，应该是根据loss估算出来的
我evaluate了x_train,y_train后发现其实际值是要低于实时演算正确率的
另外，这次更正后，又有一次准确率莫名其妙地到90%整了，而稳定值是73.6%
我现在怀疑model.h5保存再读取后，如果继续训练，可能会对模型准确率产生一定影响
不排除是由于Adam的动量项被重置的可能性，这个有待商榷
'''
from keras import backend as K
import tensorflow as tf
from keras.models import load_model


def multi_pred(y_true, y_pred):
    acc = tf.floor(K.mean(K.equal(y_true, K.round(y_pred)), axis=-1))
    return acc


'''
# 保存模型的时候与原先一样，但是读取模型由于是自定义的评价函数，需要加入custom_objects！！
model = load_model('myModel.h5', custom_objects={'multi_pred': multi_pred})
model.compile(loss='binary_crossentropy',		# binary的loss函数本身没问题
              optimizer=Adam(lr=1e-4),
              metrics=['accuracy', multi_pred],
							# 把'accuracy'去掉也可以，有的话可以直观地对比一下区别，multi_pred >= acc^3
              )
# 评分的时候，返回值的第三项才是multi_pred的值
scores = model.evaluate(x_test, y_test, verbose=0)
    print('epoch', i)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[2])	# 注意！这里改成score[2]了
'''
