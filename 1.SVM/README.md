# 支持向量机（SVM）代码实现

## 概述
本项目实现了一个基于支持向量机（SVM）的分类器，包括核函数的选择、模型的训练、预测以及展示决策边界等功能。支持多种核函数，包括线性核、RBF核、双曲正切核和多项式核，可用于解决二分类问题。

## 目录结构说明

- `Data_Mat/`：存放训练使用的数据集。
- `Output/`：存放输出结果
- `Function_Kernel.m`：核函数的实现。
- `Train_SVM.m`：训练SVM模型。
- `Predict_SVM.m`：使用训练好的SVM模型进行预测。
- `Contour_SVM.m`：展示SVM决策边界。
- `Data_Rate.m`：分隔数据集为训练集和测试集。
- `Demo_Ripley.m`与`Main_Ripley.m`：使用Ripley数据集的演示示例代码。
- `Main_SVM.m`和`Assemble_SVM.m`：SVM模型的主要执行代码。
- `qp.mexw64`：自定义的QP问题求解器，为编译后的Matlab可执行文件。

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

### SVM模型训练 `Train_SVM.m`

- **功能**：根据提供的训练数据和标签，训练SVM模型。
- **定义**：function Outputs = Train_SVM(Samples_Train, Labels_Train, C, Kernel, QPPs_Solver)
- **输入参数**：
  - `Samples_Train`：训练样本。
  - `Labels_Train`：样本对应的标签。
  - `C`：松弛变量的参数。
  - `Kernel`：核函数的类型。
  - `QPPs_Solver`：指定需要使用的QP求解器。给定`qp`为自定义求解器（qp.mexw64）；给定`QP_Matlab`为Matlab内置的求解器

### SVM模型预测 `Predict_SVM.m`

- **功能**：使用训练好的SVM模型对新的数据进行预测。
- **定义**：function [Acc, Margin, Data_Supporters, Labels_Decision, Outs_Predict] = Predict_SVM(Outs_Train, Samples_Predict, Labels_Predict)
- **输入参数**：
  - `Outs_Train`：训练过程的输出，包含训练数据、标签、核函数及松弛变量。
  - `Samples_Predict`：预测样本。
  - `Labels_Predict`：预测样本的标签。

### 决策边界展示 `Contour_SVM.m`

- **功能**：展示SVM的决策边界。
- **定义**：function [X, Y, Z] = Contour_SVM(Inputs, x_Interval, y_Interval)
- **输入参数**：
  - `Inputs`：包含训练数据、标签、核函数及松弛变量。
  - `x_Interval`：x轴的范围。
  - `y_Interval`：y轴的范围。

### 数据集分隔 `Data_Rate.m`

- **功能**：将数据集分割为训练集和测试集。
- **定义**：function [Data_SubPredict, Data_SubTrain] = Data_Rate(Data_Train, TestRate)
- **输入参数**：
  - `Data_Train`：原始训练数据集。
  - `TestRate`：测试数据的比例。

## 示例代码

- `Demo_Ripley.m`与`Main_Ripley.m`：展示如何使用Ripley数据集进行SVM模型的训练和预测。
- `Main_SVM.m`和`Assemble_SVM.m`：SVM模型的主执行文件，前者指定一个数据集，后者训练所有数据集并保存结果
