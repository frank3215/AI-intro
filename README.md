这个文件夹的内容是在[这个repo](https://github.com/ansh941/MnistSimpleCNN)和[这篇文档](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)的基础上上进行改动的。

文件一览：

```
.
./code/transforms.py
./code/modelM5
./code/modelM5/1.pth            # 训练了10个Epoch的ModelM5模型，种子为1
./code/modelM5/0.pth            # 训练了10个Epoch的ModelM5模型，种子为0
./code/backdoor_train.py        # 后门攻击的训练代码（未成功）
./code/MyModel/0.pth            # 一个MyModel模型（见论文）
./code/modelM3/1.pth            # 训练了10个Epoch的ModelM5模型，种子为1
./code/modelM3/0.pth            # 训练了10个Epoch的ModelM5模型，种子为0
./code/backdoor/1_backdoor.pth  # 尝试注入后门的模型
./code/backdoor/0_backdoor.pth  # 尝试注入后门的模型
./code/transfer_attack.py       # 通过一个参考模型进行FGSM的迁移攻击
./code/models/mymodel2.py       # MyModel2的代码（见论文）
./code/models/mymodel3.py       # MyModel3的代码（见论文）
./code/models/mymodel.py        # MyModel的代码（见论文）
./code/PGD_attack.py            # PGD攻击的代码
./code/attack.py                # 复制自 https://pytorch.org/tutorials/beginner/fgsm_tutorial.html ，稍微改了改
./code/train.py                 # 是 https://github.com/ansh941/MnistSimpleCNN 本来就有的，稍微改了改
./code/modelM7/1.pth            # 训练了10个Epoch的ModelM5模型，种子为1
./code/modelM7/0.pth            # 训练了10个Epoch的ModelM7模型，种子为0
./code/MyModel4/0.pth           # 一个MyModel4模型（见论文）
./code/MyModel3/0.pth           # 一个MyModel3模型（见论文）
./code/FGSM_attack.py           # 进行FGSM攻击
./code/backdoor_attack.py       # 尝试进行后门攻击（未）
./data                          # https://github.com/ansh941/MnistSimpleCNN 本来就有的MNIST数据未改动
```
