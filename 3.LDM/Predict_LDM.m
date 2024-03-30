function [Acc, Margin, Data_Supporters, Label_Decision, Outs_Predict] = Predict_LDM(Outs_Train, Samples_Predict, Label_Predict)

% Function:  Predicting of the LDM

%------------------- Input -------------------%
% Outs_Train includes:
%    1.Outs_Train.beta      the solution to dual problem
%   2.Outs_Train.alpha      the solution to primal problem
% 3.Outs_Train.Samples      the samples
%  4.Outs_Train.Labels      the cooresponding labels of Samples
%       5.Outs_Train.C      the parameter for slack variables
%     6.Outs_Train.Ker      kernel type

%      Samples_Predict      the samples for predicting
%       Labels_Predict      the cooresponding labels of the Samples for predictin

%------------------- Output -------------------%
%             Acc    the predicting accurate

%          Margin    includes: 
%                        Margin.Samples    the margin for every training sample
%                           Margin.MEAN    the margin mean
%                       Margin.VARIANCE    the margin variance   方差

% Data_Supporters    the support vectors
%  Label_Decision    the predicting labels
%    Outs_Predict    includes:  
%               Outs_Predict.u    the solution to the original problem    
%   Outs_Predict.Samples_Train    the samples
%          Outs_Predict.Kernel    kernel type

% Author: Wendong Wang
%  Email: d.sylan@foxmail.com
%   data: 2015,9,8


   rand('state', 2015)
   randn('state', 2015)
   

%% Main
   alpha = Outs_Train.alpha;
   Samples_Train = Outs_Train.Samples;
   G = Outs_Train.G;     %?  Outs_Trains.G?
   Labels_Train = Outs_Train.Labels;
   Kernel = Outs_Train.Ker;
   
 %------------Margin statistics------------%
   m = length(Labels_Train);
   Margin.SAMPLES = diag(Labels_Train)*G*alpha;
   Margin.MEAN = Labels_Train'*G*alpha/m;
   Margin.VARIANCE = 2*(m*alpha'*G*G*alpha-alpha'*G*Labels_Train*Labels_Train'*G*alpha)/(m^2); 
   
 %------------Search the support vectors------------%
   tau = 1e-7;
   Index = abs(alpha)>tau;
   if sum(Index)<0.5*m
       Index_Supporters = Index;
   else
       Index_Pos = find(Labels_Train==1);
       alpha_Pos = alpha(Index_Pos);
       [~, Order_Pos] = sort(abs(alpha_Pos), 'descend');
       IndexSupp_Pos = Index_Pos(Order_Pos(1:round(0.2*length(Index_Pos))));

       Index_Neg = find(Labels_Train==-1);
       alpha_Neg = alpha(Index_Neg);
       [~, Order_Neg] = sort(abs(alpha_Neg), 'descend');
       IndexSupp_Neg = Index_Neg(Order_Neg(1:round(0.2*length(Index_Neg))));
       Index_Supporters = union(IndexSupp_Pos, IndexSupp_Neg);

   end
   Samples_Supporters = Samples_Train(Index_Supporters, :);
   Labels_Supporters = Labels_Train(Index_Supporters);
   Data_Supporters = [Samples_Supporters, Labels_Supporters]; 
   
 %------------Label_Decision------------%
 % Predict the label
   Label_Decision = -ones(length(Label_Predict), 1);
   Value_Decision = Function_Kernel(Samples_Predict, Samples_Train, Kernel)*alpha;
   Label_Decision(Value_Decision>=0) = 1;
   
 %------------Acc------------%
   Acc = sum(Label_Decision==Label_Predict)/length(Label_Predict);

 %------------Outs_Predict------------%
   Outs_Predict.Samples_Train = Samples_Train;
   Outs_Predict.alpha = alpha;
   Outs_Predict.Kernel = Kernel;
 
end

