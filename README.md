# New_Wheel
This repository is used to store new wheel

# Test_Wheel
临时储存仍有一定BUG的轮子


## INDEX

### 函数
| 文件名 | 函数名 | 功能描述 | 创建人 | 最后版本时间 |
| ------ | :------: | ------ | :------: | ------ |
| load_data.py | load_data_use_csv | 从csv文件中读取标签并从相关图片目录中读取图片 | Koolo233 | 2018-11-26 |
| getscore.py | get_score | 对比预测csv和标答csv之间的差距并评出正确率 | fenghansen | 2018-11-28 |
| image_2_tfrecords.py | image_2_tfrecords | 将数据集转化为TFrecords文件，加快图片存取速度和稳定性 | fenghansen | 2018-11-26 |
| multi_pred.py | multi_pred | 支持正确地显示与评测multi-label模型的准确率acc | fenghansen | 2018-11-28 |
| weight_binary_crossentropy.py | weight_binary_crossentropy | Keras的计算具有权重的sigmoid交叉熵函数 | fenghansen | 2018-12-4 |
| call_baiduAPI.py | call_baiduAPI_POI | 调用百度地图API查询特定地点经纬度 | Koolo233 | 2018-11-29 |
| call_baiduAPI.py | call_baiduAPI_POI_batch | 调用百度地图API爬取特定地理范围内满足指定关键字的数据 | Koolo233 | 2018-11-29 |
| CAM.py | visualize_class_activation_map | 生成CAM图 | Koolo233 | 2018-12-15 |



### 类
| 文件名 | 类名 | 功能描述 | 创建人 | 最后版本时间 |
| ------ | :------: | ------ | ------ | ------ |


### 参考资料
| 文件名 | 内容描述 | 创建人 | 参考资料创建时间 |
| ------ | :------: | ------ | ------ |
| predict.py | 根据你的Keras模型和测试集来生成你的预测csv | fenghansen | 2018-11-26 |
| DivResNet.py | RoboCup初赛形状识别的基于多分类方法+精简版ResNet的Keras+TFrecords实现 | fenghansen | 2018-11-28 |

