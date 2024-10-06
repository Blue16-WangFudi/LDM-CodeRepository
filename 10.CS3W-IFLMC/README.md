# CS3W-IFLMC代码实现

## 概述
CS3W-IFLMC模型的数据代码

## 目录结构说明

- `Data_mat/`：存放训练使用的数据集。
- `Output/`：存放输出结果。
- CD_FLDM:坐标下降法求解FLDM对偶问题。
- Contour_FLDM：坐标下降法求解FLDM对偶问题
- CS3WD：基于ldm的分类任务，使用坐标下降法进行参数优化
- Data_Rate:分割数据集为训练集和测试集
- Main_FLDM:使用FLDM进行分类(主要是数据分割)
- Main_Ripley:用于执行数据加载、模型训练、参数优化、交叉验证、预测、结果统计和可视化。
- Predict_FLDM：使用训练好的线性判别机模型来预测新样本的标签，并计算模型的预测准确率以及一些额外的统计信息
- `Function_Kernel.m`：核函数的实现。
- `Fuzzy_MemberShip.m`与`IFuzzy_MemberShip.m`：计算Fuzzy与IFuzzy的数值。
- Train_FLDM:训练FLDM。

## 执行代码

- Main_Ripley
- Main_FLDM
