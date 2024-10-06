function [Acc,Label_Decision] = Predict_FTBLDM(Outs_Train, Samples_Predict, Labels_Predict,Samples_Train)




%% Main
   beta1 = Outs_Train.beta1; 
   beta2 = Outs_Train.beta2; 
   Kernel = Outs_Train.Kernel;
   K = Outs_Train.K;
   
%    b = u(end);
%    Kernel = Outs_Train.Ker; 
%    if strcmp(Kernel.Type, 'Linear')
%        beta = Outs_Train.beta;
%    end
%    Samples_Train = Outs_Train.Samples;
%    Labels_Train = Outs_Train.Labels;
%    K = Outs_Train.K;
%    
%    
%  %------------Margin statistics------------%
%    m = length(Labels_Train);
%    Margin.SAMPLES = diag(Labels_Train)*K'*u;
%    Margin.MEAN = Labels_Train'*K'*u/m;
%    Margin.VARIANCE = 2*(m*u'*K*K'*u-u'*K*Labels_Train*Labels_Train'*K'*u)/(m^2);
%    
%  % %------------Search the support vectors------------%
%    tau = 1e-7;  
%    if strcmp(Kernel.Type, 'Linear')
%        w = u(1:end-1);
%        Index_beta = abs(beta)>tau;
%        if sum(Index_beta)<0.5*m
%            Index_Supporters = Index_beta;
%        else
%            Index_Pos = find(Labels_Train==1);
%            beta_Pos = beta(Index_Pos);
%            [~, Order_Pos] = sort(abs(beta_Pos), 'descend');
%            IndexSupp_Pos = Index_Pos(Order_Pos(1:round(0.2*length(Index_Pos))));
%            
%            Index_Neg = find(Labels_Train==-1);
%            beta_Neg = beta(Index_Neg);
%            [~, Order_Neg] = sort(abs(beta_Neg), 'descend');
%            IndexSupp_Neg = Index_Neg(Order_Neg(1:round(0.2*length(Index_Neg))));
%            Index_Supporters = union(IndexSupp_Pos, IndexSupp_Neg);
%        end   
%    else
%        alpha = u(1:end-1);
%        Index_alpha = abs(alpha)>tau;
%        if sum(Index_alpha)<0.5*m
%            Index_Supporters = Index_alpha;
%        else
%            Index_Pos = find(Labels_Train==1);
%            alpha_Pos = alpha(Index_Pos);
%            [~, Order_Pos] = sort(abs(alpha_Pos), 'descend');
%            IndexSupp_Pos = Index_Pos(Order_Pos(1:round(0.2*length(Index_Pos))));
%            
%            Index_Neg = find(Labels_Train==-1);
%            alpha_Neg = alpha(Index_Neg);
%            [~, Order_Neg] = sort(abs(alpha_Neg), 'descend');
%            IndexSupp_Neg = Index_Neg(Order_Neg(1:round(0.2*length(Index_Neg))));
%            Index_Supporters = union(IndexSupp_Pos, IndexSupp_Neg);
%        end
%    end
%    Samples_Supporters = Samples_Train(Index_Supporters, :);
%    Labels_Supporters = Labels_Train(Index_Supporters);
%    Data_Supporters = [Samples_Supporters, Labels_Supporters];  
   
 %------------Label_Decision------------%
 % Predict the label
   Label_Decision = -ones(length(Labels_Predict), 1);

   distance1 = abs(Function_Kernel(Samples_Predict, Samples_Train, Kernel)*beta1)/sqrt(beta1'*K*beta1);
   distance2 = abs(Function_Kernel(Samples_Predict, Samples_Train, Kernel)*beta2)/sqrt(beta2'*K*beta2);
   Value_Decision = distance1-distance2;
   
   Label_Decision(Value_Decision<=0) = 1;
   
 %------------Acc------------%
   Acc = sum(Label_Decision==Labels_Predict)/length(Labels_Predict);
   
 
end

