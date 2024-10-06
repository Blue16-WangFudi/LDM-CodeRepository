%% Initilizing the enviroment 
   clear all
   close all
   clc
   rand('state', 2015)
   randn('state', 2015)
   
   
%% Load and prepare the data
 % Training data
   Output_Train = load('Data_mat\Ripley_Train.mat'); 
   DataTrain_Name = fieldnames(Output_Train);   % A struct data
   Data_Train = getfield(Output_Train, DataTrain_Name{1}); % Abstract the data
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
   Label_Predict  = Data_Predict (:, end);

   
%% Some public parameters
 % F_LDM Type
   FLDM_Type = 'F1_LDM';
   Kernel.Type = 'RBF';
   
   switch FLDM_Type
       case 'F1_LDM'
           if strcmp(Kernel.Type, 'Linear')
               u = 0.1;                  % OK
               lambda1 = 0.03125;        % OK
               lambda2 = 0.03125;        % OK
               C = 256;                  % OK   
               
               Value_Contour = 1; 
               Str_Legend = 'Linear F1\_LDM';
           elseif strcmp(Kernel.Type, 'RBF')
               u = 0.1;                % OK         
               lambda1 = 0.015625;       % OK 
               lambda2 = 0.015625;       % OK
               C = 100;                % OK         
               Kernel.gamma = 10.9227;      % OK
               
               Value_Contour = 1;      % OK
               Str_Legend = 'RBF-kernel F1\_LDM';
           else
               disp('Wrong parameters are provided.')
               return
           end
       case 'F2_LDM'
           if strcmp(Kernel.Type, 'Linear')
               u = 0.1;             % OK
               lambda1 = 0.5;       % OK
               lambda2 = 0.0625;    % OK
               C = 100;             % OK
               
               Value_Contour = 1;
               Str_Legend = 'Linear F2\_LDM';
           elseif strcmp(Kernel.Type, 'RBF')
               u = 0.1;             %  OK    91.9%@u = 0.5; Chose from u_Interval = linspace(0.1, 0.5, 3);  
               lambda1 = 0.0039063;    %  OK    91.9%@lambda1 = 2^(-8);      Chose from lambda1_Interval = 2.^(-8:-2);
               lambda2 = 0.0039063;    %  OK    91.9%@lambda2 = 2^(-8);      Chose from lambda2_Interval = 2.^(-8:-2); 
               C = 1;              %  OK    91.9%@C = 2^2;               Chose from C_Interval = 2.^(-8:8);  
               Kernel.gamma = 10.9068;    %  OK    91.9%@Kernel.gamma = 2^3;    Chose from gamma_Interval = 2.^(-4:4);
               
               Value_Contour =1e-2;      % Best@20150910   91.9%@Kernel.gamma = 2^3;
               Str_Legend = 'RBF-kernel F2\_LDM';
           else
               disp('Wrong parameters are provided.')
               return
           end
       otherwise
           fprintf('%g\s','  Wrong inputs are provided.');
           return
   end
   QPPs_Solver = 'CD_FLDM';
     
     
%% Train and predict  
    C_s.C = C*abs(Labels_Train);
    tic
    C_s.s = Fuzzy_MemberShip(Samples_Train, Labels_Train, Kernel, u);
    Outs_Train = Train_FLDM(Samples_Train, Labels_Train, lambda1, lambda2, C_s, FLDM_Type, Kernel, QPPs_Solver);
    t = toc;
   % Predict the data
    [Acc, Margin, Data_Supporters, Label_Decision, Outs_Predict] = Predict_FLDM(Outs_Train, Samples_Predict, Label_Predict);
   
    
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
   [X, Y, Z] = Contour_FLDM(Outs_Predict, x_Interval, y_Interval); 
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
   
    
    
   
   