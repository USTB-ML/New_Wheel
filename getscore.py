import os
import csv
# 该文件用于计算你的预测答案和标答（csv）的差距，即准确率评测。
# 大家只要改一下csv的名字就好了，调用时直接getscore(src_file,test_file)即可，给详细路径也行，给相对路径也行，随你便
# 另外，文件来自新哥
# 运行后就是下面这个样子：
# C:\ProgramData\Anaconda3\python.exe F:/python/getscore.py
# {'right': 7550, 'wrong': 450, 'not_find': 0} 0.94375

# 计分器
class scoring_one(object):
    def __init__(self, standard_dir, test_dir):
        self.standard_dir = standard_dir
        self.test_dir = test_dir
        if not os.path.exists(self.standard_dir):
            print('standard_dir not find')
            return None
        if not os.path.exists(self.test_dir):
            print(self.test_dir + ' not find')
            return None

    def points_count(self):
        right = 0
        wrong = 0
        not_find = 0
        src_dict = self.get_standard_dict(self)
        test_dict = self.get_test_dict()
        for key in src_dict:
            if self.check_key_in_dict(test_dict, key) == 0:
                not_find = not_find + 1
            elif test_dict[key] == src_dict[key]:
                right = right + 1
            else:
                wrong = wrong + 1

        return {'right': right, 'wrong': wrong, 'not_find': not_find}

    @staticmethod
    def check_key_in_dict(check_dict, key):
        if key in check_dict:
            return 1
        else:
            return 0

    @staticmethod
    def get_standard_dict(self):
        dict_club = {}
        line_flag = 0
        with open(self.standard_dir)as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if line_flag == 0:
                    line_flag = 1
                else:
                    result = row[1:4]
                    dict_club[row[0]] = result
        return dict_club

    def get_test_dict(self):
        dict_club = {}
        line_flag = 0
        with open(self.test_dir)as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if line_flag == 0:
                    line_flag = 1
                else:
                    result = row[1:4]
                    dict_club[row[0]] = result
        return dict_club


def get_score(src_file=r'myTry.csv', test_file=r'testALL.csv'):
    # 删掉的这两行是实例
    # src_file = 'F:\\python\\' + 'myTry.csv'
    # test_file = 'F:\\python\\' + 'testALL.csv'
    s = scoring_one(src_file, test_file)
    out = s.points_count()
    result = out['right'] / (out['right'] + out['wrong'] + out['not_find'])
    print(out, result)
    return out, result


get_score()
