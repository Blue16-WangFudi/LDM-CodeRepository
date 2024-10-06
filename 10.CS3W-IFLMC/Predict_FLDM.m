function [Acc, Margin, Data_Supporters, Label_Decision, Outs_Predict, Value_Decision] = Predict_FLDM(Outs_Train, Samples_Predict, Labels_Predict)

% Function:  Predicting of the FLDM

%------------------- Input -------------------%
% Outs_Train includes:
%       1.Outs_Train.u         the solution to dual problem
% 2.Outs_Train.Samples         the samples
%  3.Outs_Train.Labels         the cooresponding labels of Samples
%     4.Outs_Train.Ker         kernel type
%       5.Outs_Train.K          K used to compute the margin mean and margin variance

%      Samples_Predict         the samples for predicting
%         bels_Predict         the cooresponding labels of the Samples for predicting

%------------------- Output -------------------%
%             Acc    the predicting accurate

%          Margin    includes: 
%                        Margin.Samples    the margin for every training sample
%                           Margin.MEAN    the margin mean
%                       Margin.VARIANCE    the margin variance

% Data_Supporters    the support vectors
%  Label_Decision    the predicting labels
%    Outs_Predict    includes:  
%                        Outs_Predict.u    the solution to the original problem    
%            Outs_Predict.Samples_Train    the samples
%                   Outs_Predict.Kernel    kernel type

% Author: Wendong Wang
%  Email: d.sylan@foxmail.com
%   data: 2015,9,8
% updata: 2015,10,7


 rand('state', 2015)
 randn('state', 2015)

%% Main
   u = Outs_Train.u; 
   b = u(end);
   Kernel = Outs_Train.Ker; 
   if strcmp(Kernel.Type, 'Linear')
       beta = Outs_Train.beta;
   end
   Samples_Train = Outs_Train.Samples;
   Labels_Train = Outs_Train.Labels;
   K = Outs_Train.K;
   
   
 %------------Margin statistics------------%
   m = length(Labels_Train);
   Margin.SAMPLES = diag(Labels_Train)*K'*u;
   Margin.MEAN = Labels_Train'*K'*u/m;
   Margin.VARIANCE = 2*(m*u'*K*K'*u-u'*K*Labels_Train*Labels_Train'*K'*u)/(m^2);
   
 % %------------Search the support vectors------------%
   tau = 1e-7;  
   if strcmp(Kernel.Type, 'Linear')
       w = u(1:end-1);
       Index_beta = abs(beta)>tau;
       if sum(Index_beta)<0.5*m
           Index_Supporters = Index_beta;
       else
           Index_Pos = find(Labels_Train==1);
           beta_Pos = beta(Index_Pos);
           [~, Order_Pos] = sort(abs(beta_Pos), 'descend');
           IndexSupp_Pos = Index_Pos(Order_Pos(1:round(0.2*length(Index_Pos))));
           
           Index_Neg = find(Labels_Train==-1);
           beta_Neg = beta(Index_Neg);
           [~, Order_Neg] = sort(abs(beta_Neg), 'descend');
           IndexSupp_Neg = Index_Neg(Order_Neg(1:round(0.2*length(Index_Neg))));
           Index_Supporters = union(IndexSupp_Pos, IndexSupp_Neg);
       end   
   else
       alpha = u(1:end-1);
       Index_alpha = abs(alpha)>tau;
       if sum(Index_alpha)<0.5*m
           Index_Supporters = Index_alpha;
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
   end
   Samples_Supporters = Samples_Train(Index_Supporters, :);
   Labels_Supporters = Labels_Train(Index_Supporters);
   Data_Supporters = [Samples_Supporters, Labels_Supporters];  
   
 %------------Label_Decision------------%
 % Predict the label
   Label_Decision = -ones(length(Labels_Predict), 1);
   if strcmp(Kernel.Type, 'Linear')
       Value_Decision = Samples_Predict*w + b*abs(Labels_Predict);
   else
       Value_Decision = Function_Kernel(Samples_Predict, Samples_Train, Kernel)*alpha + b*abs(Labels_Predict);
   end
   Label_Decision(Value_Decision>=0) = 1;
   
 %------------Acc------------%
   Acc = sum(Label_Decision==Labels_Predict)/length(Labels_Predict);
   
 %------------Outs_Predict------------%
   Outs_Predict.u = u;
   Outs_Predict.Samples_Train = Samples_Train;
   Outs_Predict.Kernel = Kernel;
 
end

