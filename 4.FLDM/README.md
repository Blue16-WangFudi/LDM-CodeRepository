# 模糊大边缘分布机（FLDM）代码实现

## 概述
本项目实现了一个基于大边缘分布机（FLDM）的分类器，包括核函数的选择、模型的训练、预测以及展示决策边界等功能。支持多种核函数，包括线性核、RBF核、双曲正切核和多项式核，可用于解决二分类问题。

## 目录结构说明

- `Data_Mat/`：存放训练使用的数据集。
- `Output/`：存放输出结果。
- `Function_Kernel.m`：核函数的实现。
- `Train_FLDM.m`：训练FLDM模型。
- `Predict_FLDM.m`：使用训练好的FLDM模型进行预测。
- `Contour_FLDM.m`：展示FLDM决策边界。
- `Data_Rate.m`：分隔数据集为训练集和测试集。
- `Fuzzy_MemberShip.m`：计算Fuzzy数值。
- `Demo_Ripley.m`与`Main_Ripley.m`：使用Ripley数据集的演示示例代码。
- `Main_FLDM.m`和`Assemble_FLDM.m`：FSVM模型的主要执行代码。
- `CD_FLDM.m`：坐标下降法用于解决FLDM的对偶问题。

## 核心组件说明

### 核函数实现 `Function_Kernel.m`

- **功能**：实现不同类型的核函数，返回核矩阵。
- **定义**：function Matrix_Ker = Function_Kernel(A, B, Kernel)
- **输入参数**：
  - `A`：输入数据，样本位于行中。
  - `B`：另一组输入数据，样本位于行中，与A的样本大小相同。
  - `Kernel`：核函数的类型和参数。
- **支持的核函数类型**：`Linear kernel`, `RBF kernel`, `Sigmoid kernel`, `Polynomial kernel`。
  - `Linear kernel`：提供参数：Kernel.Type = 'Linear'
  - `RBF kernel`：提供参数：Kernel.Type = 'RBF'，Kernel.gamma
  - `Sigmoid Kernel`：提供参数：Kernel.Type = 'Sigmoid'，Kernel.gamma，Kernel.c
  - `Polynomial kernel`：提供参数：Kernel.Type = 'Polynomial'，Kernel.gamma，Kernel.c，Kernel.n

### FLDM模型训练 `Train_FLDM.m`

- **功能**：根据提供的训练数据和标签，训练FLDM模型。
- **定义**：function Outs_Train = Train_FLDM(Samples_Train, Labels_Train, lambda1, lambda2, C_s, Kernel, QPPs_Solver)
- **输入参数**：
  - `Samples_Train`：训练样本。
  - `Labels_Train`：样本对应的标签。
  - `lambda1`：边际标准差的数值。
  - `lambda2`：边际均值的数值。
  - `C_s`：松弛变量的参数。
  - `Kernel`：核函数的类型。
  - `QPPs_Solver`：指定需要使用的QP求解器。给定`qp`为自定义求解器（qp.mexw64）；给定`QP_Matlab`为Matlab内置的求解器

### FLDM模型预测 `Predict_FLDM.m`

- **功能**：使用训练好的FLDM模型对新的数据进行预测。
- **定义**：function [Acc, Margin, Data_Supporters, Labels_Decision, Outs_Predict] = Predict_SVM(Outs_Train, Samples_Predict, Labels_Predict)
- **输入参数**：
  - `Outs_Train`：训练后的输出结果。
  - `Samples_Predict`：预测样本。
  - `Labels_Predict`：预测样本的标签。

### 决策边界展示 `Contour_FLDM.m`

- **功能**：绘制等高线图，展示FLDM在不同区域的预测结果。

- **定义**：function [X, Y, Z] = Contour_FLDM(Outs_Predict, x_Interval, y_Interval)

- **输入参数**：
  - `Outs_Predict`：训练后的输出结果。
  
    Outs_Predict.Samples_Train    训练样本集。
  
    Outs_Predict.alpha    原始问题的解。
  
    Outs_Predict.Kernel    核的类型。
  
  - `x_Interval`：x轴的范围。
  
  - `y_Interval`：y轴的范围。

### 数据集分隔 `Data_Rate.m`

- **功能**：将数据集分割为训练集和测试集。
- **定义**：function [Data_SubPredict, Data_SubTrain] = Data_Rate(Data_Train, TestRate)
- **输入参数**：
  - `Data_Train`：原始训练数据集。
  - `TestRate`：测试数据的比例。
## 示例代码

- `Demo_Ripley.m`与`Main_Ripley.m`：展示如何使用Ripley数据集进行FLDM模型的训练和预测。
- `Main_FLDM.m`和`Assemble_FLDM.m`：FLDM模型的主执行文件，前者指定一个数据集，后者训练所有数据集并保存结果
