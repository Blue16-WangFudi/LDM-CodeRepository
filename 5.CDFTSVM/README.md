# CDFTSVM代码实现

## 概述
本项目实现了一个基于CDFTSVM的分类器，包括核函数的选择、模型的训练、预测以及展示决策边界等功能。支持多种核函数，包括线性核、RBF核、双曲正切核和多项式核，可用于解决二分类问题。

## 目录结构说明

- `Data_Mat/`：存放训练使用的数据集。
- `Output/`：存放输出结果。
- `Optimiser/`：QP优化器的源代码。
- `rbf_kernel.m`：rbf核函数的实现。
- `ftsvmtrain.m`：训练CDFTSVM模型。
- `ftsvmplot.m`：进行数据绘图。
- `ftsvmclass.m`：通过已经训练好的模型对数据进行预测。
- `fuzzy.m`：计算模糊隶属度。
- `distance.m`：计算一个点到另一个点的欧氏距离。
- `setup.m`：（/Optimiser中）编译qp优化器。
- `demo.m`：CDFTSVM的示例代码。

## 核心组件说明

### 核函数实现 `rbf_kernel.m`

- **功能**：实现rbf核函数。
- **定义**：function kval = rbf_kernel(u,v,rbf_sigma,varargin)

### 训练CDFTSVM `ftsvmtrain.m`

- **功能**：训练CDFTSVM。
- **定义**：function  [ftsvm_struct] = ftsvmtrain(Traindata,Trainlabel,Parameter)

### 数据绘图 `ftsvmplot.m`

- **功能**：绘制等高线图，展示FLDM在不同区域的预测结果。
- **定义**：function ftsvmplot(ftsvm_struct,Traindata,Trainlabel)

### 对数据进行预测 `ftsvmclass.m`

- **功能**：使用已经完成的模型对数据进行预测。
- **定义**：function [acc,outclass,time, fp, fn]= ftsvmclass(ftsvm_struct,Testdata,Testlabel)
### fuzzy计算 `fuzzy.m`

- **功能**：计算模糊隶属度。
- **定义**：function [sp,sn,XPnoise,XNnoise,time]=fuzzy(Xp,Xn,Parameter)

### 距离计算 `distance.m`

- **功能**：计算一个点到另一个点的欧氏距离。
- **定义**：function [d,x]=distance(xi,X)

## 官方注释

%   Version 1.0-Oct-2014  

%

%   Coordinate Descent Fuzzy Twin Support Vector for Classification

%   mainly function

%   ftsvmtrain        - training FTSVM for classification 

%   ftsvmplot         - plot 2 dimensional classification problem

%   ftsvmclass        - calculate output from input data 

%   fuzzy             - compute fuzzy  membership for train data

%   L1CD              - optimization ftsvm  with the coordinate descent  methods

%   Kernel function

%   rbf_kernel      -  RBF kernel function

%__________________________________________________________________

%

%  Author: Bin-Bin Gao (csgaobb@gmail.com)

%  Created on 2014.10.10

%  Last modified on 2015.07.16

%  Nanjing  University  

 （/Optimiser）NOTE: The CDFTSVM contains a qp Optimiser, which is from [SVM toolbox](http://www.isis.ecs.soton.ac.uk/resources/svminfo/).

## 示例代码

- `demo.m`：CDFTSVM的示例代码。
