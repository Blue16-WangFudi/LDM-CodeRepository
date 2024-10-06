function [Acc, Margin, Data_Supporters, Labels_Decision, Outs_Predict] = Predict_SVM(Outs_Train, Samples_Predict, Labels_Predict)

% Outs_Train: It contains four parts: The training data, Traing data's label,
  % Kernel, Tradeoff for slack variables
  
% Data_Predict: Data used to predict the model whose row contains the sample 

% Labels_Predict: The label for Data_Predict whose row contains the sample

% tau: To find the support vectors

% % Outs_Train：包含四个部分：训练数据，交易数据的标签，内核，权衡松弛变量    
% % Data_Predict：用于预测其行包含样本的模型的数据
% % Labels_Predict：Data_Predict的标签，其行包含示例
% % tau：查找支持向量

%    rand('state', 2015)
%    randn('state', 2015)
   
   
%% Main
 %------------Basic settings------------%
   beta = Outs_Train.beta;
   Q = Outs_Train.Q;
   Samples_Train = Outs_Train.Samples;
   Labels_Train = Outs_Train.Labels;
   C = Outs_Train.C;
   tau = C*1e-7;
   Kernel = Outs_Train.Kernel;
   e = abs(Labels_Train);
   m = length(Labels_Train);
 % Compute the b
   b_Supporters = logical((beta>tau).*(beta<C-tau));  % A number which is bigger than tau will be regarded as a positive number（大于tau的数字将被视为正数）
   Vector_b = Labels_Train(b_Supporters) - Q(b_Supporters, :)*beta.*Labels_Train(b_Supporters);%*********************************************???
   b = mean(Vector_b);
  
 %------------Search the support vectors------------%寻找支持向量
   All_Supporters = logical(beta>tau);  % A number which is bigger than tau will be regarded as a positive number（大于tau的数字将被视为正数）
   Samples_Supporters = Samples_Train(All_Supporters, :);
   Labels_Supporters = Labels_Train(All_Supporters, :);
   Data_Supporters = [Samples_Supporters, Labels_Supporters];
 
 %------------Margin statistics------------%间隔统计  %%**************************************************************************************？？？公式？？？
   Q_beta = Q*beta;
   m_leave = e'*Labels_Train;
   Margin.SAMPLES = Q_beta + b*Labels_Train;
   Margin.MEAN = e'*(Q_beta + b*Labels_Train)/m;%间隔均值
   Margin.VARIANCE = 2*(m*Q_beta'*Q_beta-Q_beta'*e*e'*Q_beta)/m^2 + 2*b^2*(1-(m_leave/m)^2) + 4*b*(m*Labels_Train'*Q_beta-m_leave*e'*Q_beta)/m^2; %方差

 %------------Label_Decision------------% 标签决策
   Labels_Decision = -ones(length(Labels_Predict), 1);
   if strcmp(Kernel.Type, 'Linear')
       Value_Decision = Samples_Predict*Samples_Train'*diag(Labels_Train)*beta+b*abs(Labels_Decision);%Value_Decision:100*1列向量
   else
       Value_Decision = Function_Kernel(Samples_Predict, Samples_Train, Kernel)*diag(Labels_Train)*beta+b*abs(Labels_Decision);
   end
   Labels_Decision(Value_Decision>=0) = 1;
   
 %------------Acc------------%准确度
   Acc = sum(Labels_Decision==Labels_Predict)/length(Labels_Predict);

 %------------Outs_Predict------------%预测输出
   Outs_Predict.Data_Train = [Samples_Train, Labels_Train];
   Outs_Predict.Kernel = Kernel;
   Outs_Predict.beta = beta;
   Outs_Predict.b = b;

end

