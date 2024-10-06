Assemble_excel_IF对比说明：
数据集：
Data_mat、Data_mat_n、Data_mat_x、batchs

IF_Final.m：进行分类器的训练和预测，同时评估其性能
mul_LDM.m：实现一个多分类器融合的线性判别模型（LDM）训练和预测过程
mul_SVM.m：
数据预处理、线性判别模型（LDM）训练、SVM（支持向量机）训练与预测，以及多分类器结果的融合



求解：
CD_FLDM.m：坐标下降法用于解决模糊线性判别模型 F_LDM的对偶问题
CD_LDM.m：坐标下降法用于解决模糊线性判别模型LDM的对偶问题


预测：
Predict_FLDM.m：
预测模糊线性判别模型（FLDM）的结果
Predict_LDM.m：
使用模糊线性判别模型（FLDM）对给定的测试样本进行预测，并计算相关的性能指标
Predict_SVM.m：
使用支持向量机（SVM）模型对给定的测试样本进行预测，并计算相关的性能指标


训练：
Train_FLDM.m：
训练FLDM模型
Train_LDM.m：
训练线性判别模型（LDM）
Train_SVM.m：
训练SVM

Plot：
Contour_FLDM.m：生成模糊线性判别模型（FLDM）的决策边界等高线图
Contour_LDM.m：生成线性判别模型（LDM）的决策边界等高线图
Contour_SVM.m：用于生成支持向量机（SVM）模型的决策边界等高线图


计算模糊隶属度：
Copy_of_Fuzzy_MemberShip.m：计算二分类问题中数据样本的模糊隶属度
Copy_of_IFuzzy_MemberShip_wang19.m
Fuzzy_MemberShip.m
Fuzzy_MemberShip_FCM.m
IFuzzy_MemberShip.m
IFuzzy_MemberShip_liang21.m
IFuzzy_MemberShip_wang19.m


其他函数：
Data_Rate.m：分割数据集
Function_Kernel.m：
计算两个数据集 A 和 B 之间的核矩阵
svdatanorm.m：
对训练数据进行归一化处理








