# GHM_Loss
 The tensorflow implementation of GHM loss include class loss and regression loss. 
 HM loss is peoposed  in "Gradient Harmonized Single-stage Detector" published on AAAI 2019 (Oral).
 
 # 1.引言：
one-stage的目标检测算法一直存在的问题是正负样本不均衡，简单和困难样本的不均衡。在one-stage算法中，负样本的数量要远远大于正样本，而且大多数负样本是简单样本（well-classified）。单个简单负样本的梯度虽然小，但是由于数量过大，会导致简单负样本主导模型的训练。在《focal loss》中通过大大降低简单样本的分类loss来平衡正负样本，但是设计的loss引入了两个需要通过实验来调整的超参数α和γ。
本篇论文从梯度的角度出发，提出gradient harmonizing mechanism（GHM）来解决样本不均衡的问题，GHM思想不仅可以应用于anchor的分类，同时也可以应用于坐标回归。

# 2.GHM（梯度均衡机制）
首先我们要定义统计对象——梯度模长（gradient norm）。考虑简单的二分类交叉熵损失函数（binary cross entropy loss）：
![](https://github.com/GXYM/GHM_Loss/tree/master/image/20190318210808415.png)
其中 p=sigmoid(x) 为模型所预测的样本类别的概率，p* 是对应的监督。则其对 x 的梯度（导数）为：
![](https://github.com/GXYM/GHM_Loss/tree/master/image/20190318210826996.png)
于是我们可以定义一个梯度模长，g=|p-p*|。
对一个交叉熵损失函数训练收敛的单阶段检测模型，样本梯度模长的分布统计如下图：
![](https://github.com/GXYM/GHM_Loss/tree/master/image/20190318210857248.png)
于是某个样本的g值大小就可以表现这个样本是简单样本还是困难样本。从一个收敛的检测模型中统计样本梯度的分布情况如下上图所示。从图中我们可以看出，与之前所想一样，简单样本的数量要远远大于困难样本。但同时也看出，一个已经收敛的模型中还是有相当数量的非常困难的样本，我们把这些非常困难的样本当作异常值（outliers），论文指出如果一个好的模型去学习这些异常样本会导致模型准确度降低。我的理解是，这些异常值就像数据的噪声一样，比如一个长得非常像狗的蛋糕，模型学习这些异常值反而会导致模型误入歧途。

基于以上现象与分析，研究者提出了梯度均衡机制，即根据样本梯度模长分布的比例，进行一个相应的标准化（normalization），使得各种类型的样本对模型参数更新有更均衡的贡献，进而让模型训练更加高效可靠。
由于梯度均衡本质上是对不同样本产生的梯度进行一个加权，进而改变它们的贡献量，而这个权重加在损失函数上也可以达到同样的效果，此研究中，梯度均衡机制便是通过重构损失函数来实现的。
为了清楚地描述新的损失函数，我们需要先定义梯度密度（gradient density）这一概念。仿照物理上对于密度的定义（单位体积内的质量），我们把梯度密度定义为单位取值区域内分布的样本数量。
首先定义梯度密度函数（Gradient density function）
![](https://github.com/GXYM/GHM_Loss/tree/master/image/20190318210924560.png)
其中gk

gk表示第k个样本的梯度，而且
![](https://github.com/GXYM/GHM_Loss/tree/master/image/20190318211028967.png)
所以梯度密度函数GD(g)就表示梯度落在区域:[g-e/2, g+e/2]
的样本数量。再定义梯度密度协调参数（gradient density harmonizing parameter） β。
![](https://github.com/GXYM/GHM_Loss/tree/master/image/20190318211435716.png)
这里乘样本数量 N，是为了保证均匀分布或只划分一个单位区域时，该权值为 1，即 loss 不变。
可以看出，梯度密度大的样本的权重会被降低，密度小的样本的权重会增加。于是把GHM的思想应用于分别应用于分类和回归上就形成了GHM-C和GHM-R。

# 3.GHM-C
把GHM应用于分类的loss上即为GHM-C，定义如下所示
![](https://github.com/GXYM/GHM_Loss/tree/master/image/20190318211455508.png)
根据GHM-C的loss计算方式，候选样本中的简单负样本和非常困难的异常样本的权重都会被降低，即loss会被降低，对于模型训练的影响也会被大大减小。正常困难样本的权重得到提升，这样模型就会更加专注于那些更为有效的正常困难样本，以提升模型的性能。GHM-C loss对模型梯度的修正效果如下图所示，横轴表示原始的梯度loss，纵轴表示修正后的。由于样本的极度不均衡，这篇论文中所有的图纵坐标都是取对数画的图。
![](https://github.com/GXYM/GHM_Loss/tree/master/image/20190318211641194.png)
结合梯度密度的分布图，和上图单个样本的的梯度修正，我们可以得出整体样本对于模型训练梯度的贡献，如下图所示。
![](https://github.com/GXYM/GHM_Loss/tree/master/image/20190318211657134.png)

 # Reference resources
https://arxiv.org/abs/1811.05181

https://github.com/libuyu/GHM_Detection

https://blog.csdn.net/u013841196/article/details/88650784

https://github.com/xyfZzz/GHM_Loss_Tensorflow

