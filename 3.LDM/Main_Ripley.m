%% Initilizing the enviroment 
   clear 
   close all
   clc
   rng('default') 
   
   
%% Load and prepare the data
 % Training data
   Output_Train = load('Data_mat\Ripley_Train.mat'); 
   DataTrain_Name = fieldnames(Output_Train);   % A struct data
   Data_Train = getfield(Output_Train, DataTrain_Name{1}); % Abstract the data
   N_Samples = size(Data_Train, 1);
   Data_Train = Data_Train(randperm(N_Samples), :);
   
 % Predicting data
   Output_Predict = load('Data_mat\Ripley_Predict.mat');
   DataPredict_Name = fieldnames(Output_Predict);   % A struct data
   Data_Predict = getfield(Output_Predict, DataPredict_Name{1}); % Abstract the data
    
   
%% Some parameters
   Kernel.Type = 'RBF'; 
   QPPs_Solver = 'CD_LDM';
   lambda1_Interval = 2.^(-8:-2);
   lambda2_Interval = 1;  
   C_Interval = [1 10 50 100 500];
   Best_C = 2*max(C_Interval);
   if strcmp(Kernel.Type, 'RBF')
       gamma_Interval = 2.^(-4:4);
       Value_Contour = 1;
       Str_Legend = 'RBF-kernel LDM';
   elseif strcmp(Kernel.Type, 'Linear')
       Value_Contour = 1;
       Str_Legend = 'Linear LDM';
   else
       disp('Wrong parameters are provided.')
       return
   end
 
  
  %% Counts
     K_fold = 5;
     switch Kernel.Type
         case 'Linear'
             Stop_Num = length(lambda1_Interval)*length(lambda2_Interval)*length(C_Interval)*K_fold + 1;
         case 'RBF'
             Stop_Num = length(lambda1_Interval)*length(lambda2_Interval)*length(C_Interval)*length(gamma_Interval)*K_fold + 1;
         otherwise
             disp('  Wrong kernel function is provided.')
             return
     end
     
     
%% Training prodecure 
    Best_Acc = 0;
    for ith_lambda1 = 1:length(lambda1_Interval)     %   lambda1
        lambda1 = lambda1_Interval(ith_lambda1);     %   lambda1
        lambda2 = lambda1;
        
%         for ith_lambda2 = 1:length(lambda2_Interval)     %   lambda2
%             lambda2 = lambda2_Interval(ith_lambda2);     %   lambda2
             
            for ith_C = 1:length(C_Interval)    %   C
                C = C_Interval(ith_C);          %   C
                
                for ith_gamma = 1:length(gamma_Interval)       %   gamma
                    
                    Indices = crossvalind('Kfold', N_Samples, K_fold); 
                    Acc_SubTrain = zeros(K_fold, 1);
                    for repeat = 1: K_fold
                        I_SubPredict = Indices==repeat;
                        I_SubTrain = ~I_SubPredict;
                        
                        Samples_SubPredict = Data_Train(I_SubPredict, 1:end-1); % The subtrain data
                        Labels_SubPredict = Data_Train(I_SubPredict, end);
                        
                        Samples_SubTrain = Data_Train(I_SubTrain, 1:end-1); % The subtrain data
                        Labels_SubTrain = Data_Train(I_SubTrain, end);
                        
                      %%%%%-------Computes the average distance between instances-------%%%%%%
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
                        SubAcc = Predict_LDM(Outs_SubTrain, Samples_SubPredict, Labels_SubPredict);
                        Acc_SubTrain(repeat) = SubAcc;
                        
                        Stop_Num = Stop_Num - 1;
                        disp([num2str(Stop_Num), ' steps remaining.'])
                        
                    end
                    
                    Index_Acc = mean(Acc_SubTrain);
                    
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
                    
                end    %  gamma
                
            end   %  C
    
%         end   % lambda2
    
    end   % lambda1
        
%% Main
  % Train the data with best parameters
    Samples_Train = Data_Train(:, 1:end-1);
    Labels_Train = Data_Train(:, end);
    tic
    Outs_Train = Train_LDM(Samples_Train, Labels_Train, Best_lambda1, Best_lambda2, Best_C*abs(Labels_Train), Best_Kernel, QPPs_Solver);
    t = toc;
  % Predict the data
    Samples_Predict = Data_Predict(:, 1:end-1);
    Labels_Predict = Data_Predict(:, end); 
    [Acc, Margin, Data_Supporters, Label_Decision, Outs_Predict] = Predict_LDM(Outs_Train, Samples_Predict, Labels_Predict);
   
   
%% Statistical results 
    disp(['  The training time is ', num2str(t), ' seconds.'])
    disp(['  The predicting accurate is ', num2str(100*Acc), '%.'])
    disp(['  The Best_lambda1 is ', num2str(Best_lambda1) '.'])
    disp(['  The Best_lambda2 is ', num2str(Best_lambda2) '.'])
    disp(['  The Best_C is ', num2str(Best_C) '.'])
    if strcmp(Best_Kernel.Type, 'RBF')
        disp(['  The Best_gamma is ', num2str(Best_Kernel.gamma) '.'])
    end
    Margin_MEAN = Margin.MEAN;
    Str_MEAN = sprintf('  The Margin MEAN is %0.2e', Margin_MEAN);
    disp(Str_MEAN)
    Margin_VARIANCE = Margin.VARIANCE;
    Str_VARIANCE = sprintf('  The Margin VARIANCE is %0.2e', Margin_VARIANCE);
    disp(Str_VARIANCE)
    

%% Visualization
   figure(1)
   Margin_SAMPLES = Margin.SAMPLES;
   Margin_Unique = unique(Margin_SAMPLES);
   Margin_Histc = histc(Margin_SAMPLES, Margin_Unique);
   Margin_Cumsum = cumsum(Margin_Histc)/length(Labels_Train);
   plot(Margin_Unique, Margin_Cumsum, 'k');
   legend(Str_Legend)
   figure(2)
   plot(Samples_Train(Labels_Train==1, 1), Samples_Train(Labels_Train==1, 2), 'r+', 'MarkerSize',8, 'LineWidth', 2); % Positive trainging data
   hold on
   plot(Samples_Train(Labels_Train==-1, 1), Samples_Train(Labels_Train==-1, 2), 'bx', 'MarkerSize',8, 'LineWidth', 2); % Negative trainging data
   plot(Data_Supporters(:, 1), Data_Supporters(:, 2), 'ko', 'MarkerSize',8, 'LineWidth', 2);  % The support vectors
   legend('Class 1', 'Class 2', 'Support vectors')
 % The Intervals for both X and Y axise
   x_Interval = linspace(min(Samples_Train(:, 1)), max(Samples_Train(:, 1)), 100);
   y_Interval = linspace(min(Samples_Train(:, 2)), max(Samples_Train(:, 2)), 100);
 % Contours
   [X, Y, Z] = Contour_LDM(Outs_Predict, x_Interval, y_Interval); 
   [Con_Pos, h_Pos] = contour(X, Y, Z, Value_Contour*[1 1], ':', 'Color', 'k', 'LineWidth', 1);
   clabel(Con_Pos, h_Pos, 'Color','k', 'FontSize', 12, 'FontWeight', 'bold');
   [Con_Decsi, h_Decsi] = contour(X, Y, Z, [0 0], '-', 'Color', 'k', 'LineWidth', 2);  
   clabel(Con_Decsi, h_Decsi, 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
   [Con_Neg, h_Neg] = contour(X, Y, Z, Value_Contour*[-1 -1], ':', 'Color','k', 'LineWidth', 1);
   clabel(Con_Neg, h_Neg, 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
   clabel(Con_Neg, h_Neg, 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
 % Reminder
   load handel
   sound(y)


