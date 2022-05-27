# PrivateDL

[Accurate Differentially Private Deep Learning on the Edge](https://ieeexplore.ieee.org/document/9372811)

PrivateDL是一种新颖的差分隐私深度学习框架 (PrivateDL)，该框架能够有效地减少边缘端深度学习训练时本地噪声的添加以及全局噪声的添加, 进而实现同等隐私保护程度下更高的模型精度。

- [PrivateDL](#privatedl)
  - [1.介绍](#1介绍)
    - [1.1 基于抽样的灵敏度估计算法(Sampling-Based Sensitivity Estimation)](#11-基于抽样的灵敏度估计算法sampling-based-sensitivity-estimation)
    - [1.2 基于关键数据点的虚拟批次放大算法(Virtual Batch Size Amplification Via Critical Set)](#12-基于关键数据点的虚拟批次放大算法virtual-batch-size-amplification-via-critical-set)
  - [2.代码和环境配置](#2代码和环境配置)
    - [2.1环境配置](#21环境配置)
      - [2.1.1 Spark环境配置](#211-spark环境配置)
      - [2.1.2 Python环境配置](#212-python环境配置)
    - [2.2运行](#22运行)
      - [2.2.1 代码层次结构](#221-代码层次结构)
      - [2.2.2 参数介绍](#222-参数介绍)
      - [2.2.3 运行](#223-运行)
    - [2.3支持的模型与数据集](#23支持的模型与数据集)
  - [3.实验细节描述](#3实验细节描述)
    - [3.1本地噪声降噪](#31本地噪声降噪)
    - [3.2全局噪声降噪](#32全局噪声降噪)
  - [4.项目成员](#4项目成员)
    - [4.1 北京理工大学](#41-北京理工大学)

## 1.介绍

深度学习越来越多地被运用于边缘侧的本地数据训练中, 而边缘侧的深度学习要面临的一个主要问题就是如何在保证精度的情况下尽可能的防止隐私泄露。差分隐私技术作为一种常用的深度学习隐私保护方法, 其通过向数据添加特定噪声的方式来进行隐私保护, 但是差分隐私所带的边缘侧端的本地噪声以及中心服务器端的全局噪声会降低模型的准确率。而已有的边缘侧深度学习差分隐私框架注重于通过梯度裁剪的方法进行本地噪声的添加, 然而这种方法忽略了边缘侧梯度的动态变化的特性以及其对全局噪声的影响。因此本项目设计并实现了一个新的差分隐私深度学习框架 (PrivateDL)，来减少边缘端深度学习训练时本地噪声的添加以及全局噪声的添加, 进而实现同等隐私保护程度下更高的模型精度。最终通过实验表明了该框架的有效性, 与现有框架相比, PrivateDL框架在模型准确性方面有着最高5倍的提升。


PrivateDL框架的流程图如下所示：
<img src="resource/image-20220520134702643.png" alt="image-20220520134702643" width="500px" />

### 1.1 基于抽样的灵敏度估计算法(Sampling-Based Sensitivity Estimation)

该算法在数据训练时对需要计算灵敏度的数据 (例如梯度) 进行随机取样, 然后计算取样出的数据的标准差,进而通过标准差来计算数据的灵敏度。相比于裁剪法计算灵敏度, 该方法能够动态地计算数据的灵敏度。
<img src="resource/抽样法.png" width="600px" />
### 1.2 基于关键数据点的虚拟批次放大算法(Virtual Batch Size Amplification Via Critical Set)
该算法通过识别一个批次中的关键数据点来去除冗余数据, 以此来减少模型训练时对边缘节点的资源压力, 从而节省训练时间与节点资源,使得边缘端能够承受更大的batch size,进而减少全局噪声的添加。该方法基于SlimML实现，其中SlimML参见[SlimML](https://github.com/LINC-BIT/SlimML)。

## 2.代码和环境配置

### 2.1环境配置

#### 2.1.1 Spark环境配置

**1. Apache spark下载**

在浏览器输入网址[https://spark.apache.org/downloads.html](https://spark.apache.org/downloads.html)即可进入spark的下载界面,如下图所示

同时下面的链接可以选择要下载的版本[https://archive.apache.org/dist/spark/](https://archive.apache.org/dist/spark/)

![](resource/2022-05-27-18-35-31.png)


**2. spark 安装**

首先将安装包上传至需要安装的机器上，然后解压并且创建软连接

```bash
tar -xvf spark-2.4.8-bin-hadoop.tgz -C ~/opt
cd ~/opt
ln -s spark-2.4.8-bin-without-hadoop  spark
```

Spark的目录结构如下：

- bin: 可执行脚本，比如常用的spark-shell, pyspark等。
- data：示例程序使用数据
- jars：依赖的jar包
- R：R API包
- yarn：整合yarn相关内容
- conf：配置文件目录
- examples：示例程序
- kubernetes：K8S相关内容
- licenses：许可文件
- python：python API包
- sbin：管理使用的脚本，比如: start-all.sh、start-master.sh等。

**3. 配置环境变量**

使用`vim ~/.bashrc`修改.bashrc文件，在文件末尾添加Spark的路径

```bash
export SPARK_HOME=/home/XXX
export PATH=$SPARK_HOME/bin:$PATH
```
运行下面的命令使得配置生效
```bash
source ~/.bashrc
```

可以通过spark-shell展示spark是否正确安装


**4. 在其他节点配置spark**
   
在master节点安装完成后只需使用`scp`命令复制spark文件目录及.bashrc文件到其他节点即可

**5. Spark相关文件配置**

主要有 **spark-env.sh、log4j.properties、slaves文件**
**spark-env.sh**：一些环境配置
**log4j.properties**：spark在启动过程中会有大量日志信息打印出来，如果我们只想看警告或者错误，而不是一般信息可以在log4j.properties中设置
**slaves文件**：slaves文件主要作用是告诉spark集群哪些节点是工作节点worker
具体的配置细节可以参照网上的教程。

#### 2.1.2 Python环境配置

这里建议使用Anaconda来进行python环境的配置，Anaconda是一个用于科学计算的Python发行版，适用于数据分析的Python工具，也可以用在大数据和人工智能领域。Conda则用来管理包（packages）、依赖与环境（environment）的工具（可执行命令）。

**1. anaconda安装**
进入链接下载anaconda,https://www.anaconda.com/distribution/ 选择相应的版本进行安装,使用`bash XXX.sh`即可

**2. 环境变量配置**
修改profile文件
```bash
sudo vi /etc/profile
```
在文件末尾加入
```bash
#Anaconda
export PATH=$PATH:/home/software/anaconda3/bin
```

**3. 包管理相关操作**

创建虚拟环境
```
conda create -n name python=3.7
```
删除环境
```
conda remove -n name
```

激活环境
```
source activate name
```
退出环境
```
source deactivate name
```
安装第三方库
```
conda install xxx 或 pip install xxx
```

### 2.2运行

#### 2.2.1 代码层次结构

#### 2.2.2 参数介绍

#### 2.2.3 运行



### 2.3支持的模型与数据集



| 模型      | 对应数据集     |
| --------- | -------------- |
| LeNet-5   | MNIST数据集    |
| AlexNet   | Cifar-10数据集 |
| ResNet-18 | ImageNet数据集 |



## 3.实验细节描述

### 3.1本地噪声降噪

测试本地噪声降噪时，需要测试在不同隐私预算下PrivateDL与裁剪法的模型精度对比

```bash

```


实验测试了包括LeNet-5,AlexNet,ResNet-18三种网络，部分结果如下：

<img src="resource/image-20220520141713391.png" alt="image-20220520141713391" width="400px" /> <img src="resource/image-20220520141725614.png" alt="image-20220520141725614" width="400px" />


### 3.2全局噪声降噪

测试全局噪声降噪时，需要测试不同批次大小下PrivateDL与裁剪法的模型精度对比

```bash

```

实验测试了包括LeNet-5,AlexNet,ResNet-18三种网络，部分结果如下：

<img src="resource/image-20220520142517935.png" alt="image-20220520142517935" width="400px" /> <img src="resource/image-20220520142537430.png" alt="image-20220520142537430" width="400px" />



## 4.项目成员

### 4.1 北京理工大学

李东 欧阳峻彦



