# QHZJ_classification
code base for the classification

目录划分为

* images：存放了 tSNE 可视化结果
* models：存放了四个模型的代码
  * EfficientNet.py
  * ResNet.py
  * Swin.py
  * ViT.py
* pretrain_weights
  * 预训练好的权重
  * 链接: https://pan.baidu.com/s/1jkkbnXh5wo5mNPC7lpCCzA 提取码: fret 
    --来自百度网盘超级会员v4的分享
* weights
  * 训练完的权重 
  * 链接: https://pan.baidu.com/s/1w1WEzAHIz-O6zQ_sIvdlAQ 提取码: gk8i 
    --来自百度网盘超级会员v4的分享
* focal_loss.py：实现了focal loss
* my_dataset.py：实现自定义数据集
* split_test_data.py：划分第7册数据集为 TC 和 TO
* train.py：训练脚本
* test.py：在 TO 上测试，算将未见过的类别归为已有类别的概率
* tSNE.py：可视化 tSNE 脚本
* utils.py：一些函数的集合
