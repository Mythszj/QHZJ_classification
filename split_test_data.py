"""
因为test_data.npy和test_label.npy中有训练集中未见过的数据，进行删除
"""
import os
import numpy as np

# 路径（原始数据集，前5册是train，第7册是test）
train_data_path = '../QHZJ/train_data.npy'
train_label_path = '../QHZJ/train_label.npy'
test_data_path = '../QHZJ/test_data.npy'
test_label_path = '../QHZJ/test_label.npy'
# 确保存在
assert os.path.exists(train_data_path), "dataset root: {} does not exist.".format(train_data_path)
assert os.path.exists(train_label_path), "dataset root: {} does not exist.".format(train_label_path)
assert os.path.exists(test_data_path), "dataset root: {} does not exist.".format(test_data_path)
assert os.path.exists(test_label_path), "dataset root: {} does not exist.".format(test_label_path)
# 读取相应的数据
train_data = np.load(train_data_path)
train_label = np.load(train_label_path)
test_data = np.load(test_data_path)
test_label = np.load(test_label_path)

print('train_data数量：{}'.format(len(train_data)))
print('train_data类别数量：{}'.format(len(np.unique(train_label))))
print('原始test_data数量：{}'.format(len(test_data)))
print('原始test_data类别数量：{}'.format(len(np.unique(test_label))))

# 新的测试集1：闭集测试集_TC，第7册数据（只选择在训练集中出现过的类别）
new_test_data_tc = []
new_test_label_tc = []
# 新的测试集2：开集测试集_TO，第7册数据（在训练集中未出现过的类别）
new_test_data_to = []
new_test_label_to = []


# 遍历测试集，添加到 TC 和 TO
for data, label in zip(test_data, test_label):
    # TC只选择在训练集中出现过的类别
    if label in train_label:
        new_test_data_tc.append(data)
        new_test_label_tc.append(label)
    else: # TO选择在训练集中未出现过的类别
        new_test_data_to.append(data)
        new_test_label_to.append(label)

# 转换成 np.ndarray()
new_test_data_tc = np.array(new_test_data_tc) # 2618 * 128 * 128 *3
new_test_label_tc = np.array(new_test_label_tc) # 2618

new_test_data_to = np.array(new_test_data_to) # 681 * 128 * 128 * 3
new_test_label_to = np.array(new_test_label_to) # 681

# 写入新的文件
np.save('../QHZJ/new_test_data_tc.npy', new_test_data_tc)
np.save('../QHZJ/new_test_label_tc.npy', new_test_label_tc)

np.save('../QHZJ/new_test_data_to.npy', new_test_data_to)
np.save('../QHZJ/new_test_label_to.npy', new_test_label_to)

# 统计类别个数
print('test_data闭集tc数量：{}'.format(len(new_test_data_tc))) # 2618
print('test_data闭集tc类别数量：{}'.format(len(np.unique(new_test_label_tc)))) # 415
print('test_data开集to数量：{}'.format(len(new_test_data_to))) # 681
print('test_data开集to类别数量：{}'.format(len(np.unique(new_test_label_to)))) # 316