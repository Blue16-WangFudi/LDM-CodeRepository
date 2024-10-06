%% 实验说明
% 1、IF实验：无噪声实验对比准确率，多个不同比例噪声实验（表、图）
% 2、3WD实验：保存概率在out_prob文件中，设置不同代价矩阵得到不同代价（表、图）
%% Initilizing the enviroment 
   clear all
   close all
   clc
   rand('state', 2015)
   randn('state', 2015)
   
   
%% Data preparation
%    Str(1).Name = 'Data_mat\credit-g.mat';% 1000
%    Str(2).Name = 'Data_mat\breast_cancer.mat';% 277
%    Str(3).Name = 'Data_mat\heart_statlog.mat';% 270
%    Str(4).Name = 'Data_mat\German.mat'; % 1000
%    Str(5).Name = 'Data_mat\Glass.mat';% 1000
%    Str(6).Name = 'Data_mat\Ripley_Predict.mat';% 1000
%    Str(7).Name = 'Data_mat\Ionosphere.mat';% 351
%    Str(8).Name = 'Data_mat\Iris.mat';% 150
%    Str(9).Name = 'Data_mat\Pima_indians.mat';% 768
%    Str(1).Name = 'Data_mat\Sonar.mat';% 208
%    Str(1).Name = 'Data_mat\Wine.mat';% 178
%    Str(2).Name = 'Data_mat\diabetes.mat';% 768
%    Str(3).Name = 'Data_mat\clean1.mat';% 476
%    Str(14).Name = 'Data_mat\breast_w.mat';% 683


% Str(1).Name = 'Data_mat\waveform-5000_0_1.mat';% 3345
% Str(2).Name = 'Data_mat\spambase.mat';% 4601
% Str(3).Name = 'Data_mat\kr-vs-kp.mat';% 3196
% 
% Str(4).Name = 'Data_mat\train_new.mat';% 3196
% Str(5).Name = 'Data_mat\creditcard_new.mat';% 3196
% Str(1).Name = 'Data_mat\riceClassification_new.mat';% 3196
% Str(2).Name = 'Data_mat\transfusion_new.mat';% 3196
% Str(3).Name = 'Data_mat\wine_new.mat';
% Str(4).Name = 'Data_mat\heart_new.mat';
% Str(5).Name = 'Data_mat\file_new.mat';% 3196

% Str(1).Name = 'Data_mat\German.mat'; % 1000
% Str(1).Name = 'Data_mat\Iris.mat';% 150
% Str(3).Name = 'Data_mat\Wine.mat';% 178
% Str(4).Name = 'Data_mat\Pima_indians.mat';% 768
% Str(5).Name = 'Data_mat\BUPA.mat';% 768
% Str(6).Name = 'Data_mat\heart_statlog.mat';% 270
% Str(7).Name = 'Data_mat\Glass.mat';% 1000
% Str(8).Name = 'Data_mat\Sonar.mat';% 208
% Str(9).Name = 'Data_mat\clean1.mat';% 476
% Str(10).Name = 'Data_mat\cmc_0_2.mat';% 476
Str(1).Name = 'Data_mat\kr-vs-kp.mat';% 3196
Str(2).Name = 'Data_mat\creditcard_new.mat';% 3196
Str(3).Name = 'Data_mat\waveform-5000_0_1.mat';% 3345
Str(4).Name = 'Data_mat\transfusion_new.mat';% 3196
   
%% Some parameters
 % F_LDM Type
 tic
   FLDM_Type = 'F1_LDM';
   Kernel.Type = 'RBF';
   QPPs_Solver = 'CD_FLDM';
   lambda1_Interval = 2.^(-8:-2);
   lambda2_Interval = 1;
   C_Interval = [1 10 50 100 500];
   Best_C = 2*max(C_Interval);
   Best_u = 0.1;
   
   index = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S'];
   
   
   x0 = [1,1];
%    x10=0;
%    x20=0;
%    error = 0.01;
   
   switch Kernel.Type
       case 'Linear'
           if strcmp(FLDM_Type, 'F1_LDM')
               Files_Name = 'Linear(F1_LDM)';
               Value_Contour = 1;                 %%%%%%%%%%%%%%%% Linear F1_LDM
               Str_Legend = 'Linear F1\_LDM';     %%%%%%%%%%%%%%%% Linear F1_LDM
           elseif strcmp(FLDM_Type, 'F2_LDM')
               Files_Name = 'Linear(F2_LDM)';
               Value_Contour = 1;                 %%%%%%%%%%%%%%%% Linear F2_LDM
               Str_Legend = 'Linear F2\_LDM';     %%%%%%%%%%%%%%%% Linear F2_LDM
           else
               disp('Wrong parameters are provided.')
               return
           end
       case 'RBF'
           gamma_Interval = 2.^(-4:4);
           if strcmp(FLDM_Type, 'F1_LDM')
               Files_Name = 'RBF(F1_LDM)';
               Value_Contour = 1;                  %%%%%%%%%%%%%%%% RBF-kernel F1_LDM
               Str_Legend = 'RBF-kernel F1\_LDM';  %%%%%%%%%%%%%%%% RBF-kernel F1_LDM
           elseif strcmp(FLDM_Type, 'F2_LDM')
               Files_Name = 'RBF(F2_LDM)';
               Value_Contour = 1;                  %%%%%%%%%%%%%%%% RBF-kernel F2_LDM
               Str_Legend = 'RBF-kernel F2\_LDM';  %%%%%%%%%%%%%%%% RBF-kernel F2_LDM
           else
               disp('Wrong parameters are provided.')
               return
           end
       otherwise
           disp('Wrong parameters are provided.')
           return
   end
     
 
%% Counts
   N_Times = 3;
   K_fold = 5;
   switch Kernel.Type
       case 'Linear'
           Stop_Num = length(Str)*N_Times*length(lambda1_Interval)*length(lambda2_Interval)*length(C_Interval)*K_fold + 1;
       case 'RBF'
           Stop_Num = length(Str)*N_Times*length(lambda1_Interval)*length(lambda2_Interval)*length(C_Interval)*length(gamma_Interval)*K_fold + 1;
       otherwise
           disp('  Wrong kernel function is provided.')
           return
   end
   
     
%% Train and predict the data  
   Location = [cd() '\' Files_Name];
   mkdir(Location)
   for iData = 1:length(Str)
       Str_Name = Str(iData).Name;
       Output = load(Str_Name);
       Data_Name = fieldnames(Output);   % A struct data
       Data_Original = getfield(Output, Data_Name{1}); % Abstract the data
     % Normalization
       Data_Original = [mapminmax(Data_Original(:, 1:end-1)', 0, 1)', Data_Original(:, end)]; % Map the original data to value between [0, 1] by colum
       M_Original = size(Data_Original, 1);
       Data_Original = Data_Original(randperm(M_Original), :);
       
%        noise_rate = 0.1;
%        [m,n]=size(Data_Original);
%        noise_num = randperm(M_Original, fix(noise_rate*M_Original));
%        Data_Original(noise_num, n) = Data_Original(noise_num, n).*-1;
     %%%%%%%-----------------Training the best parameters-----------------%%%%%%%
       TrainRate = 0.2;       % The scale of the tuning set 
       t_Train = zeros(N_Times, 1);
       Acc_Predict = zeros(N_Times, 1);
       MarginMEAN_Train = zeros(N_Times, 1);
       MarginSTD_Train = zeros(N_Times, 1);
       CS2WD_cost = zeros(N_Times, 1);
       CS2WD_acc = zeros(N_Times, 1);
       CS2WD_pre = zeros(N_Times, 1);
       CS2WD_recall = zeros(N_Times, 1);
       CS2WD_F1 = zeros(N_Times, 1);
         
       CI2WD_cost = zeros(N_Times, 1);
       CI2WD_acc = zeros(N_Times, 1);
       CI2WD_pre = zeros(N_Times, 1);
       CI2WD_recall = zeros(N_Times, 1);
       CI2WD_F1 = zeros(N_Times, 1);
       
       CS3WD_cost = zeros(N_Times, 1);
       CS3WD_acc = zeros(N_Times, 1);
       CS3WD_pre = zeros(N_Times, 1);
       CS3WD_recall = zeros(N_Times, 1);
       CS3WD_F1 = zeros(N_Times, 1);
       
       
       
       for Times = 1: N_Times
           
           [Data_Train, Data_Predict] = Data_Rate(Data_Original, TrainRate);   % Chose 3
           
           Samples_Train = Data_Train(:, 1:end-1);
           Labels_Train = Data_Train(:, end);
           
           Best_Acc = 0;
           for ith_lambda1 = 1:length(lambda1_Interval)    % lambda1
               lambda1 = lambda1_Interval(ith_lambda1);    % lambda1
               lambda2 = lambda1;                          % lambda2
               
%                for ith_lambda2 = 1:length(lambda2_Interval)    % lambda2
%                    lambda2 = lambda2_Interval(ith_lambda2);    % lambda2
                   
                   for ith_C = 1:length(C_Interval)    %   C
                       C = C_Interval(ith_C);          %   C
                       
                       for ith_gamma = 1:length(gamma_Interval)       %   gamma
                           
                           Indices = crossvalind('Kfold', length(Labels_Train), K_fold);
                           Acc_SubPredict = zeros(K_fold, 1);
                           for repeat = 1:K_fold

                               I_SubTrain = ~(Indices==repeat);
                               Samples_SubTrain = Samples_Train(I_SubTrain, :);
                               Labels_SubTrain = Labels_Train(I_SubTrain, :);
                               
                               %%%%%%-------Computes the average distance between instances-------%%%%%%
                               M_Sub = size(Samples_SubTrain, 1);
                               Index_Sub = nchoosek(1:M_Sub, 2); % Combination
                               delta_Sub = 0;
                               Num_Sub = size(Index_Sub, 1);
                               for i = 1:Num_Sub
                                   delta_Sub = delta_Sub + norm(Samples_SubTrain(Index_Sub(i, 1), :)-Samples_SubTrain(Index_Sub(i, 2),:), 2)/Num_Sub;
                               end
                               %%%%%%-------Computes the average distance between instances-------%%%%%%
                               Kernel.gamma = delta_Sub*gamma_Interval(ith_gamma);  %   gamma
                               
                               C_s.C = C*abs(Labels_SubTrain);
                               C_s.s = IF_new_new(Samples_SubTrain, Labels_SubTrain, Kernel, Best_u);
%                                C_s.s = DC_IF_Fuzzy_MemberShip_New(Samples_SubTrain, Labels_SubTrain, Kernel, Best_u);
%                                C_s.s = Fuzzy_MemberShip(Samples_SubTrain, Labels_SubTrain, Kernel, Best_u);
%                                 C_s.s = IFuzzy_MemberShip(Samples_SubTrain, Labels_SubTrain, Kernel, Best_u);
                               Outs_SubTrain = Train_FLDM(Samples_SubTrain, Labels_SubTrain, lambda1, lambda2, C_s, FLDM_Type, Kernel, QPPs_Solver);
                               
                             % Subpredict
                               I_SubPredict = ~ I_SubTrain;
                               Samples_SubPredict = Samples_Train(I_SubPredict, :);
                               Labels_SubPredict = Labels_Train(I_SubPredict, :);
                               SubAcc = Predict_FLDM(Outs_SubTrain, Samples_SubPredict, Labels_SubPredict);
                               Acc_SubPredict(repeat) = SubAcc;
                               
                               Stop_Num = Stop_Num - 1;
                               disp([num2str(Stop_Num), ' step(s) remaining.'])
                               
                           end
                           
                           Index_Acc = mean(Acc_SubPredict);
                           if Index_Acc>Best_Acc
                               Best_Acc = Index_Acc;
                               Best_lambda1 = lambda1;
                               Best_lambda2 = lambda2;
                               Best_C = C;
                               Best_Kernel = Kernel;
                           end
                           Proper_Epsilon = 1e-4;
                           if abs(Index_Acc-Best_Acc)<=Proper_Epsilon && C<Best_C
                               Best_Acc = Index_Acc;
                               Best_lambda1 = lambda1;
                               Best_lambda2 = lambda2;
                               Best_C = C;
                               Best_Kernel = Kernel;
                           end
                           
                           
                       end    % gamma
                       
                   end    % C
                   
%                end    % lambda2
               
           end    % lambda1
           
           Best_Cs.C = Best_C*abs(Labels_Train);
           tic
           Best_Cs.s = IF_new_new(Samples_Train, Labels_Train, Best_Kernel, Best_u);
%            Best_Cs.s = IFuzzy_MemberShip(Samples_Train, Labels_Train, Best_Kernel, Best_u);
           Outs_Train = Train_FLDM(Samples_Train, Labels_Train, Best_lambda1, Best_lambda2, Best_Cs, FLDM_Type, Best_Kernel, QPPs_Solver);
%            t_Train(Times) = toc;
           
           Samples_Predict = Data_Predict(:, 1:end-1);
           Labels_Predict = Data_Predict(:, end);
           
           [Acc, Margin, Data_Supporters, Label_Decision, Outs_Predict, Value_Decision] = Predict_FLDM(Outs_Train, Samples_Predict, Labels_Predict);
           Acc_Predict(Times) = Acc;
           MarginMEAN_Train(Times) = Margin.MEAN;
           MarginSTD_Train(Times) = Margin.VARIANCE;
           
           [probability,A,B,~] = data_with_probability(Value_Decision,Labels_Predict,x0);  %在x10,x20附近求负对数似然函数最小值对应的A,B及Probability,n_iteration代表迭代次数
           
           data_pro = [Data_Predict probability];
%            cost_matrix_2decision = [0 6;21 0];
%            cost_matrix_3decision = [0 3 6;21 4 0];
%            cost_matrix_2decision = [0 8;21 0];
%            cost_matrix_3decision = [0 8 8;21 4 0];
           cost_matrix_2decision = [0 8;21 0];
           cost_matrix_3decision = [0 6 8;21 5 0];
           test_label_c = Labels_Predict;
           num = find(test_label_c == -1);
           test_label_c(num) = 2;
           prob = [probability 1-probability];
           
           
           if Times == 1
               prob_ = cat(2, prob, test_label_c, Label_Decision);
               prob_1 = prob_;

           else Times ~= 1
               prob_ = cat(2, prob, test_label_c, Label_Decision);
               prob_1 = cat(2, prob_1, prob_);
           end

%            xlswrite('out_prob.xlsx',prob,['sheet',num2str(Times)],'A1');
           
           [result_2way,result_3way]  =  result_CSCI2WD(test_label_c, prob, cost_matrix_2decision, cost_matrix_3decision, Label_Decision);
           t_Train(Times) = toc;
%            dec = result_2way.dec;
           CS2WD_cost(Times) = result_2way.cost(1);
           CS2WD_acc(Times) = result_2way.acc(1);
           CS2WD_pre(Times) = result_2way.pre(1);
           CS2WD_recall(Times) = result_2way.recall(1);
           CS2WD_F1(Times) = result_2way.F1(1);
           
           CI2WD_cost(Times) = result_2way.cost(2);
           CI2WD_acc(Times) = result_2way.acc(2);
           CI2WD_pre(Times) = result_2way.pre(2);
           CI2WD_recall(Times) = result_2way.recall(2);
           CI2WD_F1(Times) = result_2way.F1(2);           
           
           CS3WD_cost(Times) = result_3way.cost;
           CS3WD_acc(Times) = result_3way.acc;
           CS3WD_pre(Times) = result_3way.pre;
           CS3WD_recall(Times) = result_3way.recall;
           CS3WD_F1(Times) = result_3way.F1;
           %            bayes_risk = result_2way.bayes_risk;
           
       end
 
   
%%%%%%%---------------------Save the statiatics---------------------%%%%%%%
       Name = Str_Name(10:end-4);
       Loc_Nam = [Location, '\', Name, '.txt'];
       f = fopen(Loc_Nam, 'wt');
       fprintf(f, '%s\n', ['The average cost of CS2WD is: ', sprintf('%.4f', mean(CS2WD_cost)), '.']);
       fprintf(f, '%s\n', ['The average cost of CI2WD is: ', sprintf('%.4f', mean(CI2WD_cost)), '.']);
       fprintf(f, '%s\n', ['The average cost of CS3WD is: ', sprintf('%.4f', mean(CS3WD_cost)), '.']);
       
       fprintf(f, '%s\n', ['The average accuracy of CS2WD is: ', sprintf('%.4f', mean(CS2WD_acc)), '.']);      
       fprintf(f, '%s\n', ['The average accuracy of CI2WD is: ', sprintf('%.4f', mean(CI2WD_acc)), '.']);
       fprintf(f, '%s\n', ['The average accuracy of CS3WD is: ', sprintf('%.4f', mean(CS3WD_acc)), '.']);
       
       fprintf(f, '%s\n', ['The average precision of CS2WD is: ', sprintf('%.4f', mean(CS2WD_pre)), '.']);      
       fprintf(f, '%s\n', ['The average precision of CI2WD is: ', sprintf('%.4f', mean(CI2WD_pre)), '.']);
       fprintf(f, '%s\n', ['The average precision of CS3WD is: ', sprintf('%.4f', mean(CS3WD_pre)), '.']);      
       
       fprintf(f, '%s\n', ['The average recall of CS2WD is: ', sprintf('%.4f', mean(CS2WD_recall)), '.']);      
       fprintf(f, '%s\n', ['The average recall of CI2WD is: ', sprintf('%.4f', mean(CI2WD_recall)), '.']);
       fprintf(f, '%s\n', ['The average recall of CS3WD is: ', sprintf('%.4f', mean(CS3WD_recall)), '.']);   
       
       fprintf(f, '%s\n', ['The average F1 of CS2WD is: ', sprintf('%.4f', mean(CS2WD_F1)), '.']);      
       fprintf(f, '%s\n', ['The average F1 of CI2WD is: ', sprintf('%.4f', mean(CI2WD_F1)), '.']);
       fprintf(f, '%s\n', ['The average F1 of CS3WD is: ', sprintf('%.4f', mean(CS3WD_F1)), '.']);
       fprintf(f, '%s\n', ['The detail of CS2WD is: ', sprintf('%.1f', result_2way.CS_detail), '.']);
       fprintf(f, '%s\n', ['The detail of CI3WD is: ', sprintf('%.1f', result_2way.CI_detail), '.']);
       fprintf(f, '%s\n', ['The detail of CS3WD is: ', sprintf('%.1f', result_3way.detail), '.']);
       
       fprintf(f, '%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train)), '.']);
       fprintf(f, '%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(100*Acc_Predict)) '%.']);
       fprintf(f, '%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(100*Acc_Predict)), '.']);
       fprintf(f, '%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
       fprintf(f, '%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);
       
       fprintf(f, '%s\n', 'The Best_lambda1 is: ');
       fprintf(f, '%f\n', Best_lambda1);
       fprintf(f, '%s\n', 'The Best_lambda2 is: ');
       fprintf(f, '%f\n', Best_lambda2);
       fprintf(f, '%s\n', 'The Best_C is: ');
       fprintf(f, '%f\n', Best_C);
       if  strcmp(Best_Kernel.Type, 'RBF')
           fprintf(f, '%s\n', 'The Best_gamma is: ');
           fprintf(f, '%f\n', Best_Kernel.gamma);
       end
       fclose(f);
       Str_Name_new = Str_Name(10:end-4);
       
%       记录概率及标签
       xlswrite('out_prob.xlsx',prob_1,['sheet',num2str(iData)],'A1');
%       记录准确率
       xlswrite('out.xlsx',{Str_Name_new},'sheet1',['A',num2str(iData+1)]);
       xlswrite('out.xlsx',mean(100*Acc_Predict),'sheet1',['B',num2str(iData+1)]);
%        xlswrite('out.xlsx',std(100*Acc_Predict),'sheet1',['C',num2str(iData+1)]);
       
       
%        开始记录细节
       xlswrite('out.xlsx',{Str_Name_new},'sheet2',[index(iData),'1']);
       xlswrite('out.xlsx',100*Acc_Predict,'sheet2',[index(iData),'2:',index(iData),num2str(N_Times+1)]);  
       
       xlswrite('out.xlsx',{Str_Name_new},'sheet3',['A',num2str(iData+1)]);
       xlswrite('out.xlsx',result_2way.CS_detail,'sheet3',['B',num2str(iData+1),':',index(5),num2str(iData+1)]);
       xlswrite('out.xlsx',result_2way.CI_detail,'sheet3',[index(7),num2str(iData+1),':',index(10),num2str(iData+1)]);
       xlswrite('out.xlsx',result_3way.detail,'sheet3',[index(12),num2str(iData+1),':',index(17),num2str(iData+1)]);
       
       xlswrite('out.xlsx',{Str_Name_new},'sheet4',['A',num2str(iData+1)]);
       xlswrite('out.xlsx',Best_C,'sheet4',['B',num2str(iData+1)]);
       xlswrite('out.xlsx',Best_lambda1,'sheet4',['C',num2str(iData+1)]);
       xlswrite('out.xlsx',Best_lambda2,'sheet4',['D',num2str(iData+1)]);
       if  strcmp(Best_Kernel.Type, 'RBF')
           xlswrite('out.xlsx',Best_Kernel.gamma,'sheet4',['E',num2str(iData+1)]);
       end
%%%%%%%---------------------Save the statiatics---------------------%%%%%%%
   end
%  % Reminder
%    load handel 
%    sound(y)
 toc
   
   