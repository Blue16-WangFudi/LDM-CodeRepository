# DBUPLDM代码实现-data

## 概述

DBUPLDM模型的数据代码

## 目录结构说明

- Balance_factor：解决二分类问题中的类别不平衡问题
- createfigure：展示不同算法在参数变化下的准确率对比
- CS_Unified_pin_ldm：训练和评估ldm模型，并返回最佳的准确率和对应的参数 C
- `Function_Kernel.m`：核函数的实现。
- `Fuzzy_MemberShip.m`与`Fuzzy_MemberShip_FCM.m`：计算Fuzzy与IFuzzy的数值。
- pin_csldm：自定义核和正则化参数的csldm分类器训练与测试数据预测。
- pin_svm：自定义核和正则化参数的svm分类器训练与测试数据预测。

## 执行代码

- main.m