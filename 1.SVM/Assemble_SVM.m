%% Initilizing the enviroment 
   clear all
   close all
   clc
   rng('default') 
  
   
% %% Data preparation
   Str(1).Name = 'Data_mat\Australian.mat';
   Str(2).Name = 'Data_mat\Breast_cancer(Original).mat';
   Str(3).Name = 'Data_mat\BreastTissue.mat';
   Str(4).Name = 'Data_mat\BUPA.mat';
   Str(5).Name = 'Data_mat\CMC.mat';
   Str(6).Name = 'Data_mat\German.mat';
   Str(7).Name = 'Data_mat\Glass.mat';
   Str(8).Name = 'Data_mat\Haberman.mat';
   Str(9).Name = 'Data_mat\Heart_c.mat';
   Str(10).Name = 'Data_mat\Heart_Statlog.mat';
   Str(11).Name = 'Data_mat\Hepatitis.mat';
   Str(12).Name = 'Data_mat\Ionosphere.mat';
   Str(13).Name = 'Data_mat\Iris.mat';
   Str(14).Name = 'Data_mat\New_thyroid.mat';
   Str(15).Name = 'Data_mat\Pima_indians.mat';
   Str(16).Name = 'Data_mat\Promoters.mat';
   Str(17).Name = 'Data_mat\Sonar.mat';
   Str(18).Name = 'Data_mat\Wine.mat';
   

%% Some parameters
   Kernel.Type = 'RBF';
   QPPs_Solver = 'qp';  % 'qp', 'QP_Matlab'
   C_Interval = [1 10 50 100 500];
   Best_C = 2*max(C_Interval);
   if strcmp(Kernel.Type, 'Linear')
       Files_Name = 'Output\Linear(SVM)';
   elseif strcmp(Kernel.Type, 'RBF')
       Files_Name = 'Output\RBF(SVM)';
       gamma_Interval = 2.^(-4:4);
   else
       disp('Wrong parameters are provided.')
       return
   end
     
 
%% Counts
     N_Times = 10;
     K_fold = 5;
   switch Kernel.Type
       case 'Linear'
           Stop_Num = length(Str)*N_Times*length(C_Interval)*K_fold + 1;
       case 'RBF'
           Stop_Num = length(Str)*N_Times*length(C_Interval)*length(gamma_Interval)*K_fold + 1;
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
     %%%%%%%-----------------Training the best parameters-----------------%%%%%%%
       TrainRate = 0.5;       % The scale of the tuning set 
       t_Train = zeros(N_Times, 1);
       Acc_Predict = zeros(N_Times, 1);
       MarginMEAN_Train = zeros(N_Times, 1);
       MarginSTD_Train = zeros(N_Times, 1);
       for Times = 1: N_Times
           
           [Data_Train, Data_Predict] = Data_Rate(Data_Original, TrainRate);   % Chose 3
           
           Samples_Train = Data_Train(:, 1:end-1);
           Labels_Train = Data_Train(:, end);
           
           Best_Acc = 0;
           for ith_C = 1:length(C_Interval)    %   C
               C = C_Interval(ith_C);          %   C
               
               for ith_gamma = 1:length(gamma_Interval)       %   gamma
                   
                   Indices = crossvalind('Kfold', length(Labels_Train), K_fold);
                   Acc_SubPredict = zeros(K_fold, 1);
                   for repeat = 1:K_fold
                       % SubTrain
                       
                       I_SubTrain = ~(Indices==repeat);
                       Samples_SubTrain = Samples_Train(I_SubTrain, :);
                       Labels_SubTrain = Labels_Train(I_SubTrain, :);
                       
                       %%%%%%-------Computes the average distance between instances-------%%%%%%
                       M_Sub = size(Samples_SubTrain, 1);
                       Index_Sub = nchoosek(1:M_Sub, 2); % nchoosek  两两组合
                       delta_Sub = 0;
                       Num_Sub = size(Index_Sub, 1);
                       for i = 1:Num_Sub
                           delta_Sub = delta_Sub + norm(Samples_SubTrain(Index_Sub(i, 1), :)-Samples_SubTrain(Index_Sub(i, 2),:), 2)/Num_Sub;  %样本点之间的平均间隔
                       end
                       %%%%%%-------Computes the average distance between instances-------%%%%%%
                       Kernel.gamma = delta_Sub*gamma_Interval(ith_gamma);  %   gamma
                       
                       Outs_SubTrain =  Train_SVM(Samples_SubTrain, Labels_SubTrain, C*abs(Labels_SubTrain), Kernel, QPPs_Solver);   %得到w,b用于预测
                       
                       % Subpredict
                       I_SubPredict = ~ I_SubTrain;
                       Samples_SubPredict = Samples_Train(I_SubPredict, :);
                       Labels_SubPredict = Labels_Train(I_SubPredict, :);
                       
                       SubAcc = Predict_SVM(Outs_SubTrain, Samples_SubPredict, Labels_SubPredict);
                       Acc_SubPredict(repeat) = SubAcc;
                       
                       Stop_Num = Stop_Num - 1;
                       disp([num2str(Stop_Num), ' step(s) remaining.'])
                       
                   end
                   
                   Index_Acc = mean(Acc_SubPredict);
                   if Index_Acc>Best_Acc
                       Best_Acc = Index_Acc;
                       Best_C = C;
                       Best_Kernel = Kernel;
                   end
                   Proper_Epsilon = 1e-4;
                   if abs(Index_Acc-Best_Acc)<=Proper_Epsilon && C<Best_C
                       Best_Acc = Index_Acc;
                       Best_C = C;
                       Best_Kernel = Kernel;
                   end
                   
               end    %  gamma
               
           end   %  C
           
           
           % Train with the best parameters
           tic
           Outs_Train = Train_SVM(Samples_Train, Labels_Train, Best_C*abs(Labels_Train), Best_Kernel, QPPs_Solver);
           t_Train(Times) = toc;
           
           Samples_Predict = Data_Predict(:, 1:end-1);
           Labels_Predict = Data_Predict(:, end);
           [Acc, Margin, Data_Supporters, Label_Decision, Outs_Predict] = Predict_SVM(Outs_Train, Samples_Predict, Labels_Predict);
           Acc_Predict(Times) = Acc;
           MarginMEAN_Train(Times) = Margin.MEAN;
           MarginSTD_Train(Times) = Margin.VARIANCE;
       end
        
%%%%%%%---------------------Save the statiatics---------------------%%%%%%%
       Name = Str_Name(10:end-4);
       Loc_Nam = [Location, '\', Name, '.txt'];
       f = fopen(Loc_Nam, 'wt');
       fprintf(f, '%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train)), '.']);
       fprintf(f, '%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(100*Acc_Predict)) '%.']);
       fprintf(f, '%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(100*Acc_Predict)), '.']);
       fprintf(f, '%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
       fprintf(f, '%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);
       
       fprintf(f, '%s\n', 'The Best_C is: ');
       fprintf(f, '%f\n', Best_C);
       if  strcmp(Best_Kernel.Type, 'RBF')
           fprintf(f, '%s\n', 'The Best_gamma is: ');
           fprintf(f, '%f\n', Best_Kernel.gamma);
       end
       fclose(f);
   end
 % Reminder
   load handel 
   sound(y)
    
   
   