"""
此文件为百度API调用方法集成，
目前有下述功能：
call_baiduAPI_POI: 用于传入特定地点查询经纬度，返回一个地点
call_baiduAPI_POI_batch：用于传入搜索关键字查询经纬度，返回一批数据


History:

++++++++++++++++++++++++++++create+++++++++++++++++++++++++
Author: Koolo233 <Koolo233@163.com>
Created: 2018-11-24
++++++++++++++++++++++++++++update+++++++++++++++++++++++++
Author:
"""

# encoding: utf-8
from urllib.request import urlopen, quote
import pandas as pd
import json
import sys
import os


def call_baiduAPI_POI(address, key, whether_limit=None, whether_check=True):
    """
    :param address: 查询关键字
    :param key: 百度密匙
    :param whether_limit:是否加入限定词，如 查询”XXX“， 加入限定词”武汉“ ，则会查询“武汉的XXX”， 默认为不加入
    :param whether_check: 是否检测地址
                            True：print检测的地点
                            Fasle：不输出
                            默认为True
    :return: 返回由经纬度 lat, lng 组成的元组
    """

    if whether_check:
        print(address)

    url = 'http://api.map.baidu.com/geocoder/v2/'
    output = 'json'
    ak = key  # 浏览器端密钥
    if whether_limit:
        address = whether_limit + address
        address = quote(address)
    else:
        address = quote(address)

    uri = url + '?' + 'address=' + address + '&output=' + output + '&ak=' + ak
    req = urlopen(uri)
    res = req.read().decode() 
    temp = json.loads(res)
    lat = temp['result']['location']['lat']
    lng = temp['result']['location']['lng']
    return_data = (lat, lng)
    return return_data


def call_baiduAPI_batch(coordi_lower_left, coordi_top_right, nb_slice, key, search_list, search_nb_every_batch,
                        save_name, whether_print_data=True, whether_print_processing=True):
    """
    :param coordi_lower_left: 搜索区域左下角经纬度
    :param coordi_top_right: 搜索区域右上角经纬度
    :param nb_slice: 切片数目，如设定为2时，则会将区域分为2*2四个部分进行搜索
    :param key: 百度API密匙
    :param search_list: 搜索关键字的条目，以字符串数组的形式传入, 将会对每一个关键进行搜索
    :param search_nb_every_batch: 每一区域搜索时最大返回量
    :param save_name: 保存文件路径，目前支持CSV文件保存
    :param whether_print_data: 是否监测数据
                                True：输出数据
                                False：不输出
    :param whether_print_processing: 是否检测搜索流程
                                        True：保存时打印当前批次
                                        False：不监测
    :return: 无
    """

    print('begin searching')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    left_bottom = coordi_lower_left  # 设置区域左下角坐标（百度坐标系）
    right_top = coordi_top_right  # 设置区域右上角坐标（百度坐标系）
    part_n = nb_slice  # 设置区域网格（2*2）
    url0 = 'http://api.map.baidu.com/place/v2/search?'
    x_item = (right_top[0] - left_bottom[0]) / part_n
    y_item = (right_top[1] - left_bottom[1]) / part_n
    ak = key
    n = 0  # 切片计数器

    list_search = search_list
    for w in range(len(list_search)):
        query = quote(list_search[w])  # 搜索关键词设置，转换中文
        for i in range(part_n):
            for j in range(part_n):
                left_bottom_part = [left_bottom[0] + i * x_item, left_bottom[1] + j * y_item]  # 切片的左下角坐标
                right_top_part = [right_top[0] + i * x_item, right_top[1] + j * y_item]  # 切片的右上角坐标
                for k in range(search_nb_every_batch):
                    url = url0 + 'query=' + query + '&page_size=20&page_num=' + str(k) + '&scope=1&bounds=' + str(
                        left_bottom_part[1]) + ',' + str(left_bottom_part[0]) + ',' + str(
                        right_top_part[1]) + ',' + str(right_top_part[0]) + '&output=json&ak=' + ak
                    data = urllib2.urlopen(url)
                    hjson = json.loads(data.read())
                    if hjson['message'] == 'ok':
                        results = hjson['results']
                        data = list(map(lambda m: (
                        results[m]["name"], results[m]["address"], results[m]["location"]["lat"],
                        results[m]["location"]["lng"]), range(len(results))))
                        data = pd.DataFrame(data, columns=['name', "address", "lat", "lng"])
                        data.to_csv(save_name, index=False, mode='a+', encoding='UTF-8')
                    if whether_print_data:
                        print(data)
                n += 1
                if whether_print_processing:
                    print('----------------------------------------')
                    print('第' + str(n) + '个切片入库成功')

    print('++++++++++++++++++++++++++++++++++++++++++++++++')
    print('searching over')


