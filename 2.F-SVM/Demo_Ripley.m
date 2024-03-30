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
   M_Train = size(Data_Train, 1);
%  % Normalization
%    Data_Train = [mapminmax(Data_Train(:, 1:end-1)', 0, 1)', Data_Train(:, end)]; % Map the original data to value between [0, 1] by colum
   Samples_Train = Data_Train(:, 1:end-1);
   Labels_Train = Data_Train(:, end);
   
 % Predicting data
   Output_Predict = load('Data_mat\Ripley_Predict.mat');
   DataPredict_Name = fieldnames(Output_Predict);   % A struct data
   Data_Predict = getfield(Output_Predict, DataPredict_Name{1}); % Abstract the data
%  % Normalization
%    Data_Predict = [mapminmax(Data_Predict(:, 1:end-1)', 0, 1)', Data_Predict(:, end)]; % Map the original data to value between [0, 1] by colum
   Samples_Predict = Data_Predict(:, 1:end-1);
   Labels_Predict  = Data_Predict (:, end);
    
   
%% Some parameters
   Kernel.Type = 'RBF';
   QPPs_Solver = 'qp';  % 'qp', 'QP_Matlab'

   if strcmp(Kernel.Type, 'Linear')       
       C = 1;                   % OK
     
       Value_Contour = 1;
       Str_Legend = 'Linear FSVM';
   elseif strcmp(Kernel.Type, 'RBF')           
       C = 1;                    % OK      
       Kernel.gamma = 2.7017;    % OK
     
       Value_Contour = 1;
       Str_Legend = 'RBF-kernel FSVM';
   else
       disp('Wrong parameters are provided.')
       return
   end 
   u = 0.3;

          
%% Train and predict 
  % Train the data
    tic
    s = Fuzzy_MemberShip(Samples_Train, Labels_Train, Kernel, u);
    Outs_Train = Train_FSVM(Samples_Train, Labels_Train, C*s, Kernel, QPPs_Solver);
    t = toc; 
   % Predict the data
     [Acc, Margin, Data_Supporters, Label_Decision, Outs_Predict] = Predict_FSVM(Outs_Train, Samples_Predict, Labels_Predict);
   
   
%% Statistical results 
  % Predicting accurate
    disp(['  The training time is ', num2str(t), ' seconds.'])
    disp(['  The predicting accurate is ', num2str(100*Acc), '%.'])
    Margin_MEAN = Margin.MEAN;
    Str_MEAN = sprintf('  The Margin MEAN is %0.2e', Margin_MEAN);
    disp(Str_MEAN)
    Margin_VARIANCE = Margin.VARIANCE;
    Str_VARIANCE = sprintf('  The Margin VARIANCE is %0.2e', Margin_VARIANCE);
    disp(Str_VARIANCE)

    
%% Visualization
   figure(1)
   plot(Samples_Train(Labels_Train==1, 1), Samples_Train(Labels_Train==1, 2), 'r+', 'MarkerSize',8, 'LineWidth', 2); % Positive trainging data
   hold on
   plot(Samples_Train(Labels_Train==-1, 1), Samples_Train(Labels_Train==-1, 2), 'bx', 'MarkerSize',8, 'LineWidth', 2); % Negative trainging data
   plot(Data_Supporters(:, 1), Data_Supporters(:, 2), 'ko', 'MarkerSize',8, 'LineWidth', 2);  % The support vectors
   legend('Class 1', 'Class 2', 'Support vectors')
 % The Intervals for both X and Y axise
   x_Interval = linspace(min(Samples_Train(:, 1)), max(Samples_Train(:, 1)), 100);
   y_Interval = linspace(min(Samples_Train(:, 2)), max(Samples_Train(:, 2)), 100);
 % Contours
   [X, Y, Z] = Contour_FSVM(Outs_Predict, x_Interval, y_Interval); 
   [Con_Pos, h_Pos] = contour(X, Y, Z, Value_Contour*[1 1], ':', 'Color', 'k', 'LineWidth', 1);
   clabel(Con_Pos, h_Pos, 'Color','k', 'FontSize', 12, 'FontWeight', 'bold');
   [Con_Decsi, h_Decsi] = contour(X, Y, Z, [0 0], '-', 'Color', 'k', 'LineWidth', 2);  
   clabel(Con_Decsi, h_Decsi, 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
   [Con_Neg, h_Neg] = contour(X, Y, Z, Value_Contour*[-1 -1], ':', 'Color','k', 'LineWidth', 1);
   clabel(Con_Neg, h_Neg, 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
   clabel(Con_Neg, h_Neg, 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
   figure(2)
   Margin_SAMPLES = Margin.SAMPLES;
   Margin_Unique = unique(Margin_SAMPLES);
   Margin_Histc = histc(Margin_SAMPLES, Margin_Unique);
   Margin_Cumsum = cumsum(Margin_Histc)/length(Labels_Train);
   plot(Margin_Unique, Margin_Cumsum, 'k');
   legend(Str_Legend)
   
   