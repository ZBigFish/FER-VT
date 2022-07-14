# FER-VT
The unofficial implementation of  paper "Facial expression recognition with grid-wise attention and visual transformer"  
论文Facial expression recognition with grid-wise attention and visual transformer的非官方个人实现。

## Warning
The code and parameter weights are not given in the original paper. This project is implemented by pytorch according to the method of the paper. It should be noted that after many tests, the results can not reach the accuracy mentioned in the paper, with a difference of about 10% (in the case of applied data enhancement and additional data, of course, it will be lower if not applied, reaching a maximum of 15%). The purpose of this project is only for academic exchange and publicity, and to confirm whether the above problems are caused by my work.  

---------------------------------------------------------------------------------------------------------------------------------------------------------------

原论文并未给出代码与参数权重，本项目为按照论文方法使用pytorch实现。需要注意的是，在我个人经过多次试验，结果均不能达到论文中所提到的准确度，相差约10%（在已应用数据增强和额外数据的情况下，当然不用会更低，达到最高15%）。本项目目的仅为学术的交流与公开，并确认以上问题是否是因为我的工作出现问题。

## Structure
The dataset was not added and needs to be downloaded by yourself because of the size of the file.  
Dataset: it stores the three data sets ferplus, ck+ and RAF DB used in the paper. 

I have created folders under each data set to represent the data set storage structure. And I attached the modified ferplus official data set generation code for personalized data enhancement (based on the pytorch version of my experiment, some required data enhancements can only be implemented manually, and I chose to save new files after direct enhancement,).

Ck+ and raf-db datasets can be decompressed and put directly into the corresponding structure of the folder; Ferplus datasets can be decompressed into the dataset folder and generated_ training_ data. Py for data generation.

Model: it stores the network architecture model and the weight to be saved during training.

---------------------------------------------------------------------------------------------------------------------------------------------------------------

由于文件大小缘故，dataset文件夹下数据集均未添加到仓库中，需要自行下载。  

dataset：其内存放论文中所使用的三个数据集ferplus、ck+和raf-db，数据集存放结构我已经在每个数据集下创建了文件夹来表示。

并且我附带了修改过的ferplus官方数据集生成代码以进行个性化的数据增强（基于我实验的pytorch版本，部分需要的数据增强只能手动实现，我选择直接增强后保存新的文件）。  

CK+与RAF-DB数据集可以解压对应部分后直接放入文件夹对应结构中；ferplus数据集可以解压缩包到dataset文件夹下，执行generate_training_data.py进行数据生成。

model：其内存放网络架构模型与训练时要保存的权重。

## Requirement

Python 3.6+,pytorch 1.0.0+

```Python
pip install -r requirements.txt
```

## Get Start
Make sure your data set and environment are ready, and execute the following command:

确保你的数据集和环境都已就绪，执行下面的命令：
```cmd
python train.py
```

You can modify train.py to carry out personalized training.

## Propose your code or comment
Just pull issue or request if you have a better idea or question.

I will reply after seeing it.
