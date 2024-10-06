%% Initilizing the enviroment 
   clear all
   close all
   clc
   rand('state', 2015)
   randn('state', 2015)


%% Data
    Data_Train = [
        10.44  10.29  -1
        10.69  10.52  -1
        10.82  10.31  -1
        10.70  10.65  -1
        10.31  10.84  -1
        10.42  10.58  -1
        
        11.11  11.00  -1
        
        11.31  11.68   1
        11.13  11.51   1
        11.36  11.12   1
        11.67  11.34   1
        11.21  11.21   1
        11.40  11.41   1
        11.60  11.19   1
        ];
    Data_Predict = [
        10.6  10.7  -1
        10.1  10.4  -1
        11.5  11.2   1
        11.0  11.4   1
        ];
     

%% Some public parameters
 % F_LDM Type
   FLDM_Type = 'F1_LDM';
   Kernel.Type = 'Linear';
   
   switch FLDM_Type
       case 'F1_LDM'
           if strcmp(Kernel.Type, 'Linear')
               u = 0.5;              
               lambda1 = 2^5;       
               lambda2 = lambda1;   
               C = 2^8;          
               
               Value_Contour = 0.5; 
               Str_Legend = 'Linear F1\_LDM';
           else
               u = 0.5;             
               lambda1 = 2^8;       
               lambda2 = lambda1;  
               C = 2^8;              
               Kernel.gamma = 2^2; 
               
               Value_Contour = 0.5;
               Str_Legend = 'RBF-kernel F1\_LDM';
           end
       case 'F2_LDM'
           if strcmp(Kernel.Type, 'Linear')
               u = 0.5;                
               lambda1 = 2^5;       
               lambda2 = lambda1; 
               C = 2^7;               
               
               Value_Contour = 1;
               Str_Legend = 'Linear F2\_LDM';
           else
               u = 0.1;               
               lambda1 = 2^2;       
               lambda2 = lambda1;   
               C = 2^7;                      
               Kernel.gamma = 2^1; 
               
               Value_Contour = 1;
               Str_Legend = 'RBF-kernel F2\_LDM';
           end
       otherwise
           disp('  Wrong inputs are provided.')
           return
   end
   QPPs_Solver = 'CD_FLDM';
   

%% Train amd predict
 %  Training
    Samples_Train = Data_Train(:, 1:end-1);
    Labels_Train = Data_Train(:, end);
    C_s.C = C*abs(Labels_Train);
    tic
    C_s.s = Fuzzy_MemberShip(Samples_Train, Labels_Train, Kernel, u);
    Outs_Train = Train_FLDM(Samples_Train, Labels_Train, lambda1, lambda2, C_s, FLDM_Type, Kernel, QPPs_Solver);
    t = toc;
    
 %  Predicting
    Samples_Predict = Data_Predict(:, 1:end-1);
    Labels_Predict = Data_Predict(:, end);
    [Acc, Margin_Samples, Data_Supporters, Label_Decision, Outs_Predict] = Predict_FLDM(Outs_Train, Samples_Predict, Labels_Predict);
    
     
%% Statistical results 
  % Predicting accurate
    disp(['  The training time is ', num2str(t), ' seconds.'])
    disp(['  The predicting accurate is ', num2str(100*Acc), '%.'])

    
%% Visualization
    if size(Samples_Train, 2)>2
        return
    end
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
    Margin_unique = unique(Margin_Samples);
    Margin_Histc = histc(Margin_Samples, Margin_unique);
    Margin_Cumsum = cumsum(Margin_Histc)/length(Labels_Train);
    plot(Margin_unique, Margin_Cumsum, 'k');
    legend(Str_Legend)



