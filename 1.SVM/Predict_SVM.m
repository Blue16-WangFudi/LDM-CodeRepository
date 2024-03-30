function [Acc, Margin, Data_Supporters, Labels_Decision, Outs_Predict] = Predict_SVM(Outs_Train, Samples_Predict, Labels_Predict)

% Outs_Train: It contains four parts: The training data, Traing data's label,
  % Kernel,Tradeoff for slack variables
  
% Data_Predict: Data used to predict the model whose row contains the sample 

% Labels_Predict: The label for Data_Predict whose row contains the sample

% tau: To find the support vectors

   rand('state', 2015)
   randn('state', 2015)
   
   
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
   b_Supporters = logical((beta>tau).*(beta<C-tau));  % A number which is bigger than tau will be regarded as a positive number
   Vector_b = Labels_Train(b_Supporters) - Q(b_Supporters, :)*beta.*Labels_Train(b_Supporters); %? 展开，第一个y为lable，后面两个y为单位列向量1？
   b = mean(Vector_b);
   
 %------------Search the support vectors------------%
   All_Supporters = logical(beta>tau);  % A number which is bigger than tau will be regarded as a positive number
   Samples_Supporters = Samples_Train(All_Supporters, :);
   Labels_Supporters = Labels_Train(All_Supporters, :);
   Data_Supporters = [Samples_Supporters, Labels_Supporters];
 
 %------------Margin statistics------------%
   Q_beta = Q*beta;
   m_leave = e'*Labels_Train;
   Margin.SAMPLES = Q_beta + b*Labels_Train;
   Margin.MEAN = e'*(Q_beta + b*Labels_Train)/m;
   Margin.VARIANCE = 2*(m*Q_beta'*Q_beta-Q_beta'*e*e'*Q_beta)/m^2 + 2*b^2*(1-(m_leave/m)^2) + 4*b*(m*Labels_Train'*Q_beta-m_leave*e'*Q_beta)/m^2; 

 %------------Label_Decision------------%
   Labels_Decision = -ones(length(Labels_Predict), 1);
   if strcmp(Kernel.Type, 'Linear')
       Value_Decision = Samples_Predict*Samples_Train'*diag(Labels_Train)*beta+b*abs(Labels_Decision);  %  wTx+b
   else
       Value_Decision = Function_Kernel(Samples_Predict, Samples_Train, Kernel)*diag(Labels_Train)*beta+b*abs(Labels_Decision);
   end
   Labels_Decision(Value_Decision>=0) = 1;
   
 %------------Acc------------%
   Acc = sum(Labels_Decision==Labels_Predict)/length(Labels_Predict);

 %------------Outs_Predict------------%
   Outs_Predict.Data_Train = [Samples_Train, Labels_Train];
   Outs_Predict.Kernel = Kernel;
   Outs_Predict.beta = beta;
   Outs_Predict.b = b;

end

