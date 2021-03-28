# 剪枝--通道剪枝

## 下面介绍一篇有关剪枝的论文：
Network Slimming-Learning Efficient Convolutional Networks through Network Slimming（Paper）2017年ICCV的一篇paper

### 剪枝要满足的要求：
   减小模型大小；减少运行时的内存占用；在不影响精度的同时，降低计算操作数；  

   剪掉一个通道的本质是要剪掉所有与这个通道相关的输入和输出连接关系，我们可以直接获得一个窄的网络，而不需要借用任何特殊的稀疏计算包。缩放因子扮演的是通道选择的角色，因为我们缩放因子的正则项和权重损失函数联合优化，网络自动鉴别不重要的通道，然后移除掉，几乎不影响网络的泛化性能。
![image](https://user-images.githubusercontent.com/80331072/112111624-6998d380-8bef-11eb-8bbb-7b2cb85e1497.png)

### 思路：
   利用BN层中的缩放因子γ 作为重要性因子，即γ越小，所对应的channel不太重要，就可以裁剪（pruning）。  
   至于什么样的γ 算小的呢？这个取决于我们为整个网络所有层设置的一个全局阈值，它被定义为所有缩放因子值的一个比例，比如我们将剪掉整个网络中70%的通道，那么我们先对缩放因子的绝对值排个序，然后取从小到大排序的缩放因子中70%的位置的缩放因子为阈值，通过这样做，我们就可以得到一个较少参数、运行时占内存小、低计算量的紧凑网络。  

### BN层表达式：

![image](https://user-images.githubusercontent.com/80331072/112111348-09099680-8bef-11eb-8a96-dfabe6939d3a.png)

其中的 γ为缩放因子，µB、σB由统计所得，γ和 β 均由反向传播自动优化。

# 剪枝的核心代码
```
#初始化
pruned = 0  
cfg = []  
cfg_mask = []  
for k, m in enumerate(model.modules()): 
    #当m为BN层时
    if isinstance(m, nn.BatchNorm2d):  
        weight_copy = m.weight.data.clone()  #获取γ
        mask = weight_copy.abs().gt(thre).float().cuda()  #大于阈值的置为1，小于阈值的置0，float()将bool值转换为float型
        remain_channels = torch.sum(mask)#保留的通道数
        #当通道剪枝为0时需要保存一个通道
        if  remain_channels == 0:  
            print('\r\n!please turn down the prune_ratio!\r\n')  
            remain_channels = 1  
            mask[int(torch.argmax(weight_copy.abs()))]=1  #获得绝对值最大的γ的索引，并将mask[索引]置为1
        pruned = pruned + mask.shape[0] - remain_channels  
        #保留mask中元素为1的通道
        m.weight.data.mul_(mask)  
        m.bias.data.mul_(mask)  
```

## 如何剪枝自己的模型？
### 1.定义好自己网络net  
### 2.在training.py中的vgg网络替换成自己的网络，运行main_1.py,得到的训练模型会被存在model_best.pth.tar中
### 3.加载训练模型，运行main_1.py,将剪枝后的模型存在pruned.pth.tar，剪枝的比例可以自己的要求去选择
### 4.加载剪枝模型，运行main_1.py,训练剪枝模型，将剪枝后训练模型存在model_pruning_best.pth.tar，

# 代码运行(vgg模型)
## Training
```
python main_1.py --s 0.001 --train-flag True --prune-flag FLase 
```
## Pruning
```
python main_1.py --model model_best.pth.tar --save pruned.pth.tar --percent 0.5 --train-flag FLase --prune-flag True 
```
## Retraining
```
python main_1.py --refine pruned.pth.tar --model model_pruning_best.pth.tar --epochs 40 --train-flag True --prune-flag FLase
```

# 运行结果
## Training Result
Test set ：Average loss:0.3296 ,Accuracy:9374/10000(93.74%)
## Pruning Result
layer index:3         total channel:64         remain channel:62  
layer index:6         total channel:64         remain channel:64  
layer index:10        total channel:128        remain channel:128  
layer index:13        total channel:128        remain channel:128  
layer index:17        total channel:256        remain channel:256  
layer index:20        total channel:256        remain channel:256  
layer index:23        total channel:256        remain channel:256  
layer index:26        total channel:256        remain channel:256  
layer index:30        total channel:512        remain channel:460  
layer index:33        total channel:512        remain channel:216  
layer index:36        total channel:512        remain channel:65  
layer index:39        total channel:512        remain channel:37  
layer index:43        total channel:512        remain channel:5  
layer index:46        total channel:512        remain channel:5  
layer index:49        total channel:512        remain channel:57  
layer index:52        total channel:512        remain channel:500  

## Retraining Result
Test set ：Average loss:0.2848 ,Accuracy:935410000(93.54%)

