# CS-LDM代码实现

## 概述

本项目实现了一个基于CS-LDM的分类器，包括核函数的选择、模型的训练、预测以及展示决策边界等功能。支持多种核函数，包括线性核、RBF核、双曲正切核和多项式核，可用于解决二分类问题
## 目录结构说明

- Data_mat：存放训练所使用的数据集
- Output：存放输出结果
- CD_LDM:坐标下降法求解LDM对偶问题
- Contour_LDM：坐标下降法求解LDM对偶问题
- csldm：基于ldm的分类任务，使用坐标下降法进行参数优化
- csldm_noise:添加高斯噪声进行数据增强
- Data_Rate:分割数据集为训练集和测试集
- Demo_Pipley:使用带有 RBF 核的 LDM 模型进行训练和预测
- Function_Kernel:计算两个数据集 A 和 B 之间的核矩阵
- Main_LDM:使用LDM进行分类(主要是数据分割）
- Main_Ripley:用于执行一个完整的机器学习任务，包括数据加载、模型训练、参数优化、交叉验证、预测、结果统计和可视化。
- Predict_LDM：使用训练好的线性判别机模型来预测新样本的标签，并计算模型的预测准确率以及一些额外的统计信息
- svdatanorm:对训练数据 X 进行归一化处理，以适应不同的核函数 ker
- Train_LDM:训练LDM