'''
此函数功能以及计算方式基本与binary_crossentropy差不多,但是加上了权重的功能,是计算具有权重的sigmoid交叉熵函数
Keras里没有这个东西，很难受，只能自己写，有可能还有bug，慎用
主要用于multi-label问题正例反例数量严重不平均的情况，比如112120个肺片，14类，有的类只有270个，那么反例就有112120-270个……
其实还有一个隐患，就是tf.nn.weighted_cross_entropy_with_logits这个函数只把pos_weight加到了前面，所以每个标签的权重就不一样了，原来正例少的样本现在的损失会放大到特别大，相应的梯度也会特别大
'''


def weight_binary_crossentropy(y_true, y_pred):
    _epsilon = tf.convert_to_tensor(1e-7, y_pred.dtype.base_dtype)
    output = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    # output = tf.log(output / (1 - output))
    # return K.sum(tf.nn.sigmoid_cross_entropy_with_logits(y_true, output), axis=-1)
    return K.sum(tf.nn.weighted_cross_entropy_with_logits(targets=y_true, logits=output, pos_weight=pos_w), axis=-1)


'''
# 保存模型的时候与原先一样，但是读取模型由于是自定义的评价函数，需要加入custom_objects！！
model = load_model('myModel.h5', custom_objects={'multi_pred': multi_pred})
model.compile(loss='binary_crossentropy',		# binary的loss函数本身没问题
              optimizer=Adam(lr=1e-4),
              metrics=['accuracy', multi_pred],    # 把'accuracy'去掉也可以，有的话可以直观地对比一下区别，multi_pred >= acc^3
              )
# 评分的时候，返回值的第三项才是multi_pred的值
    scores = model.evaluate(x_test, y_test, verbose=0)
    mean_acc = ("%.4f" % scores[1])
    multi_acc = ("%.4f" % scores[2])
    print('epoch', i)
    print('Test loss:', scores[0])
    print('Test accuracy:', mean_acc, multi_acc)
'''
