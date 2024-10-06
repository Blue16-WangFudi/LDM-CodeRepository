%% Initilizing the enviroment 
   clear all
   close all
   clc
   rand('state', 2015)
   randn('state', 2015)
   
%% Some parameters
   N_Times = 3;  %需要自己修改
%    cost_matrix_2decision = [0 8;21 0];
%    cost_matrix_3decision = [0 6 8;21 5 0];
   cost_matrix_2decision = [0 21;41 0];
%    cost_matrix_3decision = [0 6 21;41 6 0];

   cost_num = 10;  %设置需要遍历到多少代价
   theta = 0.1;  %设置边界代价系数

   CS2WD_cost_ = zeros(cost_num, 1);
   CI2WD_cost_ = zeros(cost_num, 1);
   CS3WD_cost_ = zeros(cost_num, 1);
   orgin_cost_ = zeros(cost_num, 1);
   
%% Cost martix
%    for Boundary_cost = 1: cost_num
%        cost_matrix_3decision_ = [0 Boundary_cost 8;21 Boundary_cost 0];
%        if Boundary_cost == 1
%            cost_matrix_3decision_total = cost_matrix_3decision_;
%        else
%            cost_matrix_3decision_total = cat(1, cost_matrix_3decision_total, cost_matrix_3decision_);  %每两行一个代价矩阵 
%        end
%    end
   
%% Begin to calculate
   [prob_total] = xlsread('out_prob.xlsx', 1);
   for Boundary_cost = 1: cost_num
       cost_matrix_3decision = [0 Boundary_cost 21;41 Boundary_cost 0];
       CS2WD_cost = zeros(N_Times, 1);
       CI2WD_cost = zeros(N_Times, 1);
       CS3WD_cost = zeros(N_Times, 1);
       orgin_cost = zeros(N_Times, 1);
       for Times = 1: N_Times
           prob = prob_total(:,4*Times-3:4*Times-2);
           test_label_c = prob_total(:,4*Times-1);
           Label_Decision = prob_total(:,4*Times);
           [result_2way,result_3way]  =  result_CSCI2WD(test_label_c, prob, cost_matrix_2decision, cost_matrix_3decision, Label_Decision);

           CS2WD_cost(Times) = result_2way.cost(1);
           CI2WD_cost(Times) = result_2way.cost(2);         
           CS3WD_cost(Times) = result_3way.cost;
           orgin_cost(Times) = result_3way.cost_orgin;

       end
       CS2WD_cost_mean = mean(CS2WD_cost);
       CI2WD_cost_mean = mean(CI2WD_cost);
       CS3WD_cost_mean = mean(CS3WD_cost);
       orgin_cost_mean = mean(orgin_cost);
       
       CS2WD_cost_(Boundary_cost) = CS2WD_cost_mean;
       CI2WD_cost_(Boundary_cost) = CI2WD_cost_mean;     
       CS3WD_cost_(Boundary_cost) = CS3WD_cost_mean;
       orgin_cost_(Boundary_cost) = orgin_cost_mean;
   end
%% Save the statiatics
   xlswrite('cost_experiment.xlsx',CS2WD_cost_,'sheet1','B1');
   xlswrite('cost_experiment.xlsx',CI2WD_cost_,'sheet1','C1');
   xlswrite('cost_experiment.xlsx',CS3WD_cost_,'sheet1','A1');
   xlswrite('cost_experiment.xlsx',orgin_cost_,'sheet1','D1');
       
       