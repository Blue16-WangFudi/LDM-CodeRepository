DC-IFTBLDM代码实现
存放数据集：Data_mat     Data_mat_n      Data_mat_x
Assemble_excel_IF对比:

选择不同的分类器并调整它们的超参数进行训练和回归预测：
All、All_NV_large_cross、All_NV、All_NV_large_cross、
All_NV_large_cross_mul.m、All_NV_large_cross_Times.m、
All_NV_large_cross_Times_large.m、
All_NV_large_cross_Times_usps.m、
添加噪声：
All_NV_large_cross_Times_noise.m
All_NV_noise.m


主要功能函数：

分割数据集：
Data_Rate.m:分割数据集为子预测集和子训练集


计算隶属度:
Copy_of_DC_Fuzzy_MemberShip.m、
Copy_of_IFuzzy_MemberShip_wang19.m、
DC_Fuzzy_MemberShip.m、
DC_Fuzzy_MemberShip_aaaa.
fuzzy.m
Fuzzy_MemberShip.m、
IFuzzy_MemberShip.m、
no_IFuzzy_MemberShip.m、


模型训练：
Copy_of_IFtbsvmtrain_wang19.m（训练一个FTSVM模型）
DC_IFtbsvmtrain_New.m（训练一个模糊时间序列支持向量机（FTSVM）模型）
IFtbsvmtrain.m、
IFtbsvmtrain_liang21.m、
IFtbsvmtrain_wang19.m、
membership.m:计算训练集中每个样本的模糊隶属度 s，并将其分为两个类别
tbsvmtrain.m：训练一个双隶属度支持向量机（FTSVM）模型
Train_TSVM.m：
训练一种特殊类型的三重支持向量机（TSVM），这种SVM模型旨在处理两个不同类别的数据（正类和负类），并且可能包含未标记的数据。
Train_TLDM.m：
训练一种被称为三重线性判别模型（TLDM）
Train_TBLDM.m：
训练一种被称为双隶属度线性判别模型（TBLDM）的分类器
Train_SVM.m：训练一个支持向量机（SVM）模型
Train_FTSVM.m：
训练一种被称为模糊隶属度支持向量机（FTSVM）的分类器
Train_FTBLDM.m：
训练一种被称为模糊隶属度双线性判别模型（FTBLDM）的分类器。
Train_FLDM.m：训练一种被称为模糊线性判别模型（FLDM）的分类器


对训练数据进行分类：
ftbsvmclass.m（使用训练好的模糊时间序列支持向量机（FTSVM）模型对测试数据进行分类，并计算分类的准确性。）
ftbsvmtrain.m（通过训练数据集来训练一个FTSVM模型，并返回模型的相关信息，可以用于后续的预测或分析）



对模型进行预测：
Predict_FLDM.m（用于预测分类标签，并计算一些额外的统计信息，包括准确度、边缘统计数据、支持向量等）
Predict_FTBLDM.m：（用于基于训练得到的模型参数预测一组新样本的分类标签，并计算预测的准确度）
Predict_FTSVM.m：（用于基于训练得到的模型参数预测一组新样本的分类标签，并计算预测的准确度。）
Predict_LDM.m：（用于基于训练得到的模型参数预测一组新样本的分类标签，并计算预测的准确度以及提供其他相关统计信息。）
Predict_SVM.m：（用于基于训练得到的支持向量机（SVM）模型参数来预测一组新样本的分类标签，并计算预测的准确度以及提供其他相关统计信息。）
Predict_TBLDM.m、Predict_TLDM.m、Predict_TSVM.m


分割数据集：
Data_Rate.m:分割数据集为子预测集和子训练集

其他函数：
Function_Kernel.m：计算核矩阵
rbf_kernel.m：将输入数据隐式映射到高维特征空间，使得原本线性不可分的数据在该空间中变得线性可分。
svdatanorm.m：用于对训练数据 X 进行归一化处理，以便于在使用支持向量机（SVM）或其他机器学习算法时，数据能够在核函数 ker 下的特征空间中有更好的表现。
tbsvmclass.m：测试一个已经训练好的模糊支持向量机（FTSVM）模型在测试数据上的性能。
