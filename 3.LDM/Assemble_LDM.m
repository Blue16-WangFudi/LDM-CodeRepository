%% Initilizing the enviroment 
   clear
   close all
   clc
   rng('default') 
  
   
%% Data preparation
   Str(1).Name = 'Data_mat\Promoters.mat';
   Str(2).Name = 'Data_mat\BreastTissue.mat';
%    Str(3).Name = 'Data_mat\BUPA.mat';
%    Str(4).Name = 'Data_mat\Heart_c.mat';
%    Str(5).Name = 'Data_mat\Hepatitis.mat';
%    Str(6).Name = 'Data_mat\Ionosphere.mat';
%    Str(7).Name = 'Data_mat\Leaf.mat';
%    Str(8).Name = 'Data_mat\Pima_indians.mat';
%    Str(9).Name = 'Data_mat\PLRX.mat';
%    Str(10).Name = 'Data_mat\Australian.mat';
%    Str(11).Name = 'Data_mat\Sonar.mat';

   
%% Some parameters
   Kernel.Type = 'RBF';  
   QPPs_Solver = 'CD_LDM';
   lambda1_Interval = 2.^(-8:-2);   %0.0039    0.0078    0.0156    0.0313    0.0625    0.1250    0.2500
   lambda2_Interval = 1;
   C_Interval = [1 10 50 100 500];
   if strcmp(Kernel.Type, 'Linear')
       Files_Name = 'Output\Linear(LDM)';
   elseif strcmp(Kernel.Type, 'RBF')
       Files_Name = 'Output\RBF(LDM)';
       gamma_Interval = 2.^(-4:4);
   else
       disp('Wrong parameters are provided.')
       return
   end
     
 
%% Counts
   N_Times = 2;   %原来是10
   K_fold = 2;       %原来是5
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
     %%%%%%%-----------------Training the best parameters-----------------%%%%%%%
       TrainRate = 0.5;       % The scale of the tuning set 
       t_Train = zeros(N_Times, 1);
       Acc_Predict = zeros(N_Times, 1);
       MarginMEAN_Train = zeros(N_Times, 1);
       MarginSTD_Train = zeros(N_Times, 1);
    for Times = 1: N_Times 
        
        [Data_Train, Data_Predict] = Data_Rate(Data_Original, TrainRate);   % Chose 3   TrainRate:测试集比例
        
        Samples_Train = Data_Train(:, 1:end-1);
        Labels_Train = Data_Train(:, end);
        
        Best_Acc = 0; 
       for ith_lambda1 = 1:length(lambda1_Interval)    % lambda1
           lambda1 = lambda1_Interval(ith_lambda1);    % lambda1
           lambda2 = lambda1;                          % lambda2
           
%            for ith_lambda2 = 1:length(lambda2_Interval)    % lambda2
%                lambda2 = lambda2_Interval(ith_lambda2);    % lambda2
             
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
                         
                           Outs_SubTrain = Train_LDM(Samples_SubTrain, Labels_SubTrain, lambda1, lambda2, C*abs(Labels_SubTrain), Kernel, QPPs_Solver);
                           
                         % Subpredict
                           I_SubPredict = ~ I_SubTrain;
                           Samples_SubPredict = Samples_Train(I_SubPredict, :);
                           Labels_SubPredict = Labels_Train(I_SubPredict, :);
                       
                           SubAcc = Predict_LDM(Outs_SubTrain, Samples_SubPredict, Labels_SubPredict);
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
               
%            end    % lambda2
             
       end    % lambda1
       
           tic
           Outs_Train = Train_LDM(Samples_Train, Labels_Train, Best_lambda1, Best_lambda2, Best_C*abs(Labels_Train), Best_Kernel, QPPs_Solver);
           t_Train(Times) = toc;
           
           Samples_Predict = Data_Predict(:, 1:end-1);
           Labels_Predict = Data_Predict(:, end);
           
           [Acc, Margin, Data_Supporters, Label_Decision, Outs_Predict] = Predict_LDM(Outs_Train, Samples_Predict, Labels_Predict);
           Acc_Predict(Times) = Acc;
           MarginMEAN_Train(Times) = Margin.MEAN;
           MarginSTD_Train(Times) = Margin.VARIANCE;
           
    end
%%%%%%%-----------------Predict the best parameters-----------------%%%%%%%
 
   
%%%%%%%---------------------Save the statiatics---------------------%%%%%%%
       Name = Str_Name(10:end-4);
       Loc_Nam = [Location, '\', Name, '.txt'];
       f = fopen(Loc_Nam, 'wt');
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
%%%%%%%---------------------Save the statiatics---------------------%%%%%%%
   end
%  % Reminder
%    load handel 
%    sound(y)
    
   
   