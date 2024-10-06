%% Initilizing the enviroment 
   clear all
   close all
   clc
   rand('state', 2015)
   randn('state', 2015)
   
   
%% Load and prepare the data
%    Output = load('Data_mat\Australian.mat');
%    Output = load('Data_mat\BreastTissue.mat');
%    Output = load('Data_mat\BUPA.mat');
%    Output = load('Data_mat\Heart_c.mat');
%    Output = load('Data_mat\Hepatitis.mat');
%    Output = load('Data_mat\Ionosphere.mat');
%    Output = load('Data_mat\Leaf.mat');
%    Output = load('Data_mat\Pima_indians.mat');
%    Output = load('Data_mat\PLRX.mat');
   Output = load('Data_mat\Promoters.mat');
%    Output = load('Data_mat\Sonar.mat');
    Data_Name = fieldnames(Output);   % A struct data
    Data_Original = getfield(Output, Data_Name{1}); % Abstract the data
  % Normalization
    Data_Original = [mapminmax(Data_Original(:, 1:end-1)', 0, 1)', Data_Original(:, end)]; % Map the original data to value between [0, 1] by colum
    M_Original = size(Data_Original, 1);
    Data_Original = Data_Original(randperm(M_Original), :);
 
    
%% Some parameters
 % F_LDM Type
   FLDM_Type = 'F2_LDM';
   Kernel.Type = 'Linear';
   QPPs_Solver = 'CD_FLDM';
   lambda1_Interval = 2.^(-8:-2);
   lambda2_Interval = 1;
   C_Interval = [1 10 50 100 500];
   Best_C = 2*max(C_Interval);
   Best_u = 0.1;
   switch Kernel.Type
       case 'Linear'
           if strcmp(FLDM_Type, 'F1_LDM')
               Value_Contour = 1;               %%%%%%%%%%%%%%%% Linear F1_LDM
               Str_Legend = 'Linear F1\_LDM';   %%%%%%%%%%%%%%%% Linear F1_LDM
           elseif strcmp(FLDM_Type, 'F2_LDM')
               Value_Contour = 1;               %%%%%%%%%%%%%%%% Linear F2_LDM
               Str_Legend = 'Linear F2\_LDM';   %%%%%%%%%%%%%%%% Linear F2_LDM
           else
               disp('Wrong parameters are provided.')
               return
           end
       case 'RBF'
           gamma_Interval = 2.^(-4:4);
           if strcmp(FLDM_Type, 'F1_LDM')
               Value_Contour = 1;                  %%%%%%%%%%%%%%%% RBF-kernel F1_LDM
               Str_Legend = 'RBF-kernel F1\_LDM';  %%%%%%%%%%%%%%%% RBF-kernel F1_LDM
           elseif strcmp(FLDM_Type, 'F2_LDM')
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
     N_Times = 10;
     K_fold = 5;
     switch Kernel.Type
         case 'Linear'
             Stop_Num = N_Times*length(lambda1_Interval)*length(lambda2_Interval)*length(C_Interval)*K_fold + 1;
         case 'RBF'
             Stop_Num = N_Times*length(lambda1_Interval)*length(lambda2_Interval)*length(C_Interval)*length(gamma_Interval)*K_fold + 1;
         otherwise
             disp('  Wrong kernel function is provided.')
             return
     end
     
     
%% Training prodecure 
    TrainRate = 0.5;       % The scale of the train set 
    t_Train = zeros(N_Times, 1);
    Acc_Predict = zeros(N_Times, 1);
    Acc_Leader = 0;
    MarginMEAN_Train = zeros(N_Times, 1);
    MarginSTD_Train = zeros(N_Times, 1);
    for Times = 1: N_Times

        [Data_Train, Data_Predict] = Data_Rate(Data_Original, TrainRate);   % Chose 3
        
        Samples_Train = Data_Train(:, 1:end-1);
        Labels_Train = Data_Train(:, end);
        
        Best_Acc = 0;
        for ith_lambda1 = 1:length(lambda1_Interval)    % lambda1
            lambda1 = lambda1_Interval(ith_lambda1);    % lambda1
            lambda2 = lambda1;
            
%             for ith_lambda2 = 1:length(lambda2_Interval)    % lambda2
%                 lambda2 = lambda2_Interval(ith_lambda2);    % lambda2
                
                for ith_C = 1:length(C_Interval)    %   C
                    C = C_Interval(ith_C);          %   C
                    
%                     for ith_gamma = 1:length(gamma_Interval)       %   gamma
                        
                        Indices = crossvalind('Kfold', length(Labels_Train), K_fold);
                        Acc_SubPredict = zeros(K_fold, 1);
                        for repeat = 1:K_fold
                          % SubTrain
                            I_SubTrain = ~(Indices==repeat);
                            Samples_SubTrain = Samples_Train(I_SubTrain, :);
                            Labels_SubTrain = Labels_Train(I_SubTrain, :);
                           
%                           %%%%%%-------Computes the average distance between instances-------%%%%%%
%                             M_Sub = size(Samples_SubTrain, 1);
%                             Index_Sub = combntns(1:M_Sub, 2); % Combination
%                             delta_Sub = 0;
%                             Num_Sub = size(Index_Sub, 1);
%                             for i = 1:Num_Sub
%                                 delta_Sub = delta_Sub + norm(Samples_SubTrain(Index_Sub(i, 1), :)-Samples_SubTrain(Index_Sub(i, 2),:), 2)/Num_Sub;
%                             end
%                           %%%%%%-------Computes the average distance between instances-------%%%%%%
%                             Kernel.gamma = delta_Sub*gamma_Interval(ith_gamma);  %   gamma  
                            
                          % SubTrain
                            C_s.C = C*abs(Labels_SubTrain);
                            C_s.s = Fuzzy_MemberShip(Samples_SubTrain, Labels_SubTrain, Kernel, Best_u);
                            Outs_SubTrain = Train_FLDM(Samples_SubTrain, Labels_SubTrain, lambda1, lambda2, C_s, FLDM_Type, Kernel, QPPs_Solver);
                            
                          % Subpredict
                            I_SubPredict = ~ I_SubTrain;
                            Samples_SubPredict = Samples_Train(I_SubPredict, :); % The subtrain data
                            Labels_SubPredict = Labels_Train(I_SubPredict, :);
                            SubAcc = Predict_FLDM(Outs_SubTrain, Samples_SubPredict, Labels_SubPredict);
                            Acc_SubPredict(repeat) = SubAcc;
                            
                            Stop_Num = Stop_Num - 1;
                            disp([num2str(Stop_Num), ' steps remaining.'])
                            
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
                        
                        
%                     end    %  gamma
                    
                end    %  C
            
%         end    %lambda2
            
        end    %lambda1
            
      % Train with the best parameters
        Best_Cs.C = Best_C*abs(Labels_Train);
        tic
        Best_Cs.s = Fuzzy_MemberShip(Samples_Train, Labels_Train, Best_Kernel, Best_u);
        Outs_Train = Train_FLDM(Samples_Train, Labels_Train, Best_lambda1, Best_lambda2, Best_Cs, FLDM_Type, Best_Kernel, QPPs_Solver);
        t_Train(Times) = toc;
        
        Samples_Predict = Data_Predict(:, 1:end-1);
        Labels_Predict = Data_Predict(:, end);
        [Acc, Margin, Data_Supporters, Label_Decision, Outs_Predict] = Predict_FLDM(Outs_Train, Samples_Predict, Labels_Predict);
        Acc_Predict(Times) = Acc;        
        MarginMEAN_Train(Times) = Margin.MEAN;
        MarginSTD_Train(Times) = Margin.VARIANCE;
        
        if Acc>Acc_Leader
            Acc_Leader = Acc;
            lambda1_Leader = Best_lambda1;
            lambda2_Leader = Best_lambda2;
            C_Leader = Best_C;
            Kernel_Leader = Best_Kernel;
            Margin_Leader = Margin; 
            Samples_Leader = Samples_Train;
            Labels_Leader = Labels_Train;
            Supporters_Leader = Data_Supporters;
            Outs_Leader = Outs_Predict;
        end
    end
   
   
%% Statistical results 
    disp(['  The MEAN training time is ', num2str(mean(t_Train)), ' seconds.'])
    disp(['  The MEAN predicting accurate is ', num2str(100*mean(Acc_Predict)), '%.'])
    disp(['  The STD predicting accurate is ', num2str(100*std(Acc_Predict)), '.'])
    disp(['  The lambda1_Leader is ', num2str(lambda1_Leader), '.'])
    disp(['  The lambda2_Leader is ', num2str(lambda2_Leader), '.'])
    disp(['  The C_Leader is ', num2str(C_Leader), '.'])
    if strcmp(Kernel.Type, 'RBF')
        disp(['  The gamma_Leader is ', num2str(Kernel_Leader.gamma), '.'])        
    end
    disp(['  The Acc_Leader is ', num2str(100*Acc_Leader), '%.'])
    Str_MEAN = sprintf('  The Margin MEAN is %0.2e', mean(MarginMEAN_Train));
    disp(Str_MEAN)
    Str_VARIANCE = sprintf('  The Margin VARIANCE is %0.2e', mean(MarginSTD_Train));
    disp(Str_VARIANCE)
   % Reminder
    load handel
    sound(y)
    
    
%% Visualization
 % Margin 
   figure(1)
   Margin_SAMPLES = Margin_Leader.SAMPLES;
   Margin_Unique = unique(Margin_SAMPLES);
   Margin_Histc = histcounts(Margin_SAMPLES, Margin_Unique);
   Margin_Cumsum = cumsum(Margin_Histc)/length(Labels_Train);
   plot(Margin_Unique, Margin_Cumsum, 'k');
   legend(Str_Legend, 1)
 % Data show
    if size(Samples_Leader, 2)>2
        return
    end
    figure(2)
    plot(Samples_Leader(Labels_Leader==1, 1), Samples_Leader(Labels_Leader==1, 2), 'r+', 'MarkerSize',8, 'LineWidth', 2); % Positive trainging data
    hold on
    plot(Samples_Leader(Labels_Leader==-1, 1), Samples_Leader(Labels_Leader==-1, 2), 'bx', 'MarkerSize',8, 'LineWidth', 2); % Negative trainging data
    plot(Supporters_Leader(:, 1), Supporters_Leader(:, 2), 'ko', 'MarkerSize',8, 'LineWidth', 2);  % The support vectors
    legend('Class 1', 'Class 2', 'Support vectors', 'Location','NorthEast')
 % The Intervals for both X and Y axise
    x_Interval = linspace(min(Samples_Leader(:, 1)), max(Samples_Leader(:, 1)), 100);
    y_Interval = linspace(min(Samples_Leader(:, 2)), max(Samples_Leader(:, 2)), 100);
  % Contours
    [X, Y, Z] = Contour_FLDM(Outs_Leader, x_Interval, y_Interval); 
    [Con_Pos, h_Pos] = contour(X, Y, Z, Value_Contour*[1 1], ':', 'Color', 'k', 'LineWidth', 1);
    clabel(Con_Pos, h_Pos, 'Color','k', 'FontSize', 12, 'FontWeight', 'bold');
    [Con_Decsi, h_Decsi] = contour(X, Y, Z, [0 0], '-', 'Color', 'k', 'LineWidth', 2);
    clabel(Con_Decsi, h_Decsi, 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
    [Con_Neg, h_Neg] = contour(X, Y, Z, Value_Contour*[-1 -1], ':', 'Color','k', 'LineWidth', 1);
    clabel(Con_Neg, h_Neg, 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
    
    