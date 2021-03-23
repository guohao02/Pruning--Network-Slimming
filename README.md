# 剪枝--通道剪枝
剪枝要满足的要求：减小模型大小；减少运行时的内存占用；在不影响精度的同时，降低计算操作数；

剪掉一个通道的本质是要剪掉所有与这个通道相关的输入和输出连接关系，我们可以直接获得一个窄的网络(Figure 1)，而不需要借用任何特殊的稀疏计算包。缩放因子扮演的是通道选择的角色，因为我们缩放因子的正则项和权重损失函数联合优化，网络自动鉴别不重要的通道，然后移除掉，几乎不影响网络的泛化性能。
![image](https://user-images.githubusercontent.com/80331072/112111624-6998d380-8bef-11eb-8bbb-7b2cb85e1497.png)

# 下面介绍一篇有关剪枝的论文：
Network Slimming-Learning Efficient Convolutional Networks through Network Slimming（Paper）2017年ICCV的一篇paper

思路：利用batch normalization中的缩放因子γ 作为重要性因子，即γ越小，所对应的channel不太重要，就可以裁剪（pruning）。

BN层：![image](https://user-images.githubusercontent.com/80331072/112111348-09099680-8bef-11eb-8a96-dfabe6939d3a.png)


# Training
RUN main.py --s 0.001  --epochs 160  --refine ''(空）
# Pruning
RUN prune.py --model model_best.pth.tar --save pruned.pth.tar --percent 0.5
# Retraining
RUN main.py -refine pruned.pth.tar --epochs 40
# Issues
剪枝比例过高容易出现通道数为0，需要在剪枝的是的时候添加约束条件。
