%% Initilizing the enviroment 
   clear all
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

   
%% Some public parameters
 % F_LDM Type
   Best_Kernel.Type = 'RBF';
   QPPs_Solver = 'CD_LDM';
   if strcmp(Best_Kernel.Type, 'Linear')       
       Best_lambda1 = 2^(-8);             % OK
       Best_lambda2 = 2^(-8);             % OK 
       Best_C = 10^(0);                  % OK      
       Value_Contour = 1;            % OK 
       Str_Legend = 'Linear LDM';
   elseif strcmp(Best_Kernel.Type, 'RBF')            
       Best_lambda1 = 2^(-8);            % OK 
       Best_lambda2 = 2^(-8);            % OK
       Best_C = 10^(0);                  % OK
       Best_Kernel.gamma = 2.7017;       % OK
       Value_Contour = 1;           % OK
       Str_Legend = 'RBF-kernel LDM';
   else
       disp('Wrong parameters are provided.')
       return
   end 
   
  
   
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
    Margin_MEAN = Margin.MEAN;
    Str_MEAN = sprintf('  The Margin MEAN is %0.2e', Margin_MEAN);
    disp(Str_MEAN)
    Margin_VARIANCE = Margin.VARIANCE;
    Str_VARIANCE = sprintf('  The Margin VARIANCE is %0.2e', Margin_VARIANCE);
    disp(Str_VARIANCE)
    

%% Visualization(可视化)
   figure(1)    %这是个什么东东
   Margin_SAMPLES = Margin.SAMPLES;
   Margin_Unique = unique(Margin_SAMPLES);
   Margin_Histc = histc(Margin_SAMPLES, Margin_Unique);
   Margin_Cumsum = cumsum(Margin_Histc)/length(Labels_Train);
   plot(Margin_Unique, Margin_Cumsum, 'k');
   legend(Str_Legend)
   figure(2)   %选用数据集为2维，平面图
   plot(Samples_Train(Labels_Train==1, 1), Samples_Train(Labels_Train==1, 2), 'r+', 'MarkerSize',8, 'LineWidth', 2); % Positive trainging data
   hold on
   plot(Samples_Train(Labels_Train==-1, 1), Samples_Train(Labels_Train==-1, 2), 'bx', 'MarkerSize',8, 'LineWidth', 2); % Negative trainging data
   plot(Data_Supporters(:, 1), Data_Supporters(:, 2), 'ko', 'MarkerSize',8, 'LineWidth', 2);  % The support vectors
   legend('Class 1', 'Class 2', 'Support vectors')
   
 % The Intervals for both X and Y axise（X 轴和 Y 轴的间隔）
   x_Interval = linspace(min(Samples_Train(:, 1)), max(Samples_Train(:, 1)), 100);
   y_Interval = linspace(min(Samples_Train(:, 2)), max(Samples_Train(:, 2)), 100);
   
 % Contours（轮廓）
   [X, Y, Z] = Contour_LDM(Outs_Predict, x_Interval, y_Interval); 
   [Con_Pos, h_Pos] = contour(X, Y, Z, Value_Contour*[1 1], ':', 'Color', 'k', 'LineWidth', 1);   %等高线（正类）
   clabel(Con_Pos, h_Pos, 'Color','k', 'FontSize', 12, 'FontWeight', 'bold');   %为等高线图添加高程标签
   [Con_Decsi, h_Decsi] = contour(X, Y, Z, [0 0], '-', 'Color', 'k', 'LineWidth', 2);     %等高线（超平面）
   clabel(Con_Decsi, h_Decsi, 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
   [Con_Neg, h_Neg] = contour(X, Y, Z, Value_Contour*[-1 -1], ':', 'Color','k', 'LineWidth', 1);%登高线（负类）
   clabel(Con_Neg, h_Neg, 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
   clabel(Con_Neg, h_Neg, 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
%  % Reminder
%    load handel
%    sound(y)
    
    
   
   