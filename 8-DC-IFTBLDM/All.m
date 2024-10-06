%% Initilizing the enviroment
clear all
close all
clc

%% Data preparation
%    Str(1).Name = 'Data_mat\Australian.mat';
%    Str(2).Name = 'Data_mat\Breast_cancer(Original).mat';
%    Str(3).Name = 'Data_mat\Statlog_Heart.mat';
%    Str(4).Name = 'Data_mat\BUPA.mat';
%    Str(5).Name = 'Data_mat\CMC.mat';
%    
%    Str(6).Name = 'Data_mat\German.mat';
%    Str(7).Name = 'Data_mat\Glass.mat';
%    Str(8).Name = 'Data_mat\Haberman.mat';
%    Str(9).Name = 'Data_mat\Ripley_Train';
%    Str(10).Name = 'Data_mat\Ripley_Predict.mat';
%    Str(11).Name = 'Data_mat\Hepatitis.mat';
%    
%    Str(12).Name = 'Data_mat\Ionosphere.mat';
%    Str(13).Name = 'Data_mat\Iris.mat';
%    Str(14).Name = 'Data_mat\New_thyroid.mat';
%    Str(15).Name = 'Data_mat\Pima_indians.mat';
%    Str(16).Name = 'Data_mat\Promoters.mat';
%    Str(17).Name = 'Data_mat\Sonar.mat';
%    Str(18).Name = 'Data_mat\Wine.mat';

% Str(1).Name = 'Data_mat_x\breast-w.mat';
% Str(2).Name = 'Data_mat_x\clean1.mat';
% Str(3).Name = 'Data_mat_x\cmc_0_1.mat';
% Str(4).Name = 'Data_mat_x\cmc_0_2.mat';
% Str(5).Name = 'Data_mat_x\cmc_1_2.mat';
% Str(6).Name = 'Data_mat_x\credit_a.mat';
% Str(7).Name = 'Data_mat_x\credit-g.mat';
% Str(1).Name = 'Data_mat_x\cylinder-bands.mat';
% Str(2).Name = 'Data_mat_x\diabetes.mat';
% Str(10).Name = 'Data_mat_x\heart-statlog.mat';
% Str(13).Name = 'Data_mat_x\kr-vs-kp.mat';
% Str(14).Name = 'Data_mat_x\mushroom.mat';
% Str(15).Name = 'Data_mat_x\spambase.mat';
Str(16).Name = 'Data_mat_x\tic-tac-toe.mat';
% Str(3).Name = 'Data_mat_x\waveform-5000_0_1.mat';
% Str(4).Name = 'Data_mat_x\waveform-5000_0_2.mat';

%%  Choose the Classifier
% case 1:FTBLDM and TBLDM; 2:TLDM; 3:TBSVM; 4:FTSVM and TSVM;
for type=1:1
    switch type
       case 1   %%FTBLDM and TBLDM
            %% Some parameters
            % F_LDM Type
            FLDM_Type = 'F2_LDM';
            Kernel.Type = 'RBF';
            QPPs_Solver = 'QP_Matlab';
            gamma_Interval = 2.^(-6: 6);
            lambda1_Interval = 2.^(-6: 6);
            lambda2_Interval = 2.^(-6: 6);
            C1_Interval = 2.^(-6: 6);
            C3_Interval = 2.^(-6: 6);

            
            Best_u = 0.1;
            
            
            %% Counts
            N_Times = 1;
            K_fold = 5;
            switch Kernel.Type
                case 'Linear'
                    Stop_Num = 7*N_Times*length(lambda1_Interval)*length(lambda2_Interval)*length(C1_Interval)*length(C3_Interval) + 1;
                case 'RBF'
                    Stop_Num = 1*N_Times*length(lambda1_Interval)*length(lambda2_Interval)*length(C1_Interval)*length(C3_Interval)*length(gamma_Interval) + 1;
                otherwise
                    disp('  Wrong kernel function is provided.')
                    return
            end
            
            
            %% Train and predict the data
            for iData = 16:16
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
                t_Train_ = zeros(N_Times, 1);
                Acc_Predict = zeros(N_Times, 1);
                Acc_Predict_ = zeros(N_Times, 1);
                MarginMEAN_Train = zeros(N_Times, 1);
                MarginSTD_Train = zeros(N_Times, 1);
                Accuracy = zeros();
                for Times = 1: N_Times
                    
                    [Data_Train, Data_Predict] = Data_Rate(Data_Original, TrainRate);   % Chose 3
                    
                    Samples_Train = Data_Train(:, 1:end-1);
                    Labels_Train = Data_Train(:, end);
                    
                    Best_Acc = 0;
                    Best_lambda1 = 0;
                    Best_lambda2 = 0;
                    Best_lambda3 = 0;
                    Best_lambda4 = 0;
                    Best_C1 = 0;
                    Best_C2 = 0;
                    Best_C3 = 0;
                    Best_C4 = 0;   
                    
                    Best_Acc_ = 0;
                    Best_lambda1_ = 0;
                    Best_lambda2_ = 0;
                    Best_lambda3_ = 0;
                    Best_lambda4_ = 0;
                    Best_C1_ = 0;
                    Best_C2_ = 0;
                    Best_C3_ = 0;
                    Best_C4_ = 0;                      
                    
                    
                    for ith_lambda1 = 1:length(lambda1_Interval)    % lambda1
                        lambda1 = lambda1_Interval(ith_lambda1);    % lambda1
                        lambda3 = lambda1;
                        for ith_lambda2 = 1:length(lambda2_Interval)
                          lambda2 = lambda2_Interval(ith_lambda2);
                          lambda4 = lambda2;

                          
                               for ith_C1 = 1:length(C1_Interval)    % lambda3
                                   C1 = C1_Interval(ith_C1);    % lambda3
                                   C2 = C1;
                                  for ith_C3 = 1:length(C3_Interval) 
                                       C3 = C3_Interval(ith_C3);    % lambda3
                                       C4 = C3;
 
                                                for ith_gamma = 1:length(gamma_Interval)       %   gamma

                                                    Indices = crossvalind('Kfold', length(Labels_Train), K_fold);%随机分成K_fold份，并打上标签
                                                    Acc_SubPredict = zeros(1, 1);
                                                    Acc_SubPredict_ = zeros(1, 1);
                                                    for repeat = 1:1

                                                        I_SubTrain = ~(Indices==repeat);%(Kfold-1)/Kfold 拿来训练
                                                        Samples_SubTrain = Samples_Train(I_SubTrain, :);
                                                        Labels_SubTrain = Labels_Train(I_SubTrain, :);

                                                        %%%%%%-------Computes the average distance between instances-------%%%%%%
                                                        M_Sub = size(Samples_SubTrain, 1);
                                                        Index_Sub = combntns(1:M_Sub, 2); % Combination
                                                        delta_Sub = 0;
                                                        Num_Sub = size(Index_Sub, 1);
                                                        for i = 1:Num_Sub
                                                            delta_Sub = delta_Sub + norm(Samples_SubTrain(Index_Sub(i, 1), :)-Samples_SubTrain(Index_Sub(i, 2),:), 2)/Num_Sub;
                                                        end
                                                        %%%%%%-------Computes the average distance between instances-------%%%%%%
                                                        Kernel.gamma = delta_Sub*gamma_Interval(ith_gamma);  %   gamma

                                                        I_SubA = Labels_SubTrain == 1;
                                                        Samples_SubA = Samples_SubTrain(I_SubA,:);
                                                        Labels_SubA = Labels_SubTrain(I_SubA);

                                                        I_SubB = Labels_SubTrain == -1;
                                                        Samples_SubB = Samples_SubTrain(I_SubB,:);                        
                                                        Labels_SubB = Labels_SubTrain(I_SubB);     

                                                        C_s.C1 = C1;
                                                        C_s.C2 = C2;
                                                        C_s.s2 = Fuzzy_MemberShip(Samples_SubB, Labels_SubB, Kernel, Best_u);
                                                        C_s.C3 = C3;
                                                        C_s.C4 = C4;
                                                        C_s.s1 = Fuzzy_MemberShip(Samples_SubA, Labels_SubA, Kernel, Best_u);                                   


                                                        Outs_SubTrain = Train_FTBLDM(Samples_SubA, Labels_SubA, Samples_SubB,Labels_SubB, Samples_SubTrain, lambda1,lambda2 , C_s, Kernel, QPPs_Solver);
                                                        Outs_SubTrain_ = Train_TBLDM(Samples_SubA, Labels_SubA, Samples_SubB,Labels_SubB, Samples_SubTrain, lambda1,lambda2 , C_s, Kernel, QPPs_Solver);
                                                        
                                                        % Subpredict
                                                        I_SubPredict = ~ I_SubTrain;% 1/Kfold 拿来预测
                                                        Samples_SubPredict = Samples_Train(I_SubPredict, :);
                                                        Labels_SubPredict = Labels_Train(I_SubPredict, :);


                                                        SubAcc = Predict_FTLDM(Outs_SubTrain, Samples_SubPredict,Labels_SubPredict, Samples_SubTrain);   
                                                        SubAcc_ = Predict_TLDM(Outs_SubTrain_, Samples_SubPredict,Labels_SubPredict, Samples_SubTrain);
                                                        
                                                        Acc_SubPredict(repeat) = SubAcc;
                                                        Acc_SubPredict_(repeat) = SubAcc_;
                                                        
                                                        Stop_Num = Stop_Num - 1;
                                                        
                                                        disp([num2str(Stop_Num), ' step(s) remaining.'])

                                                    end

                                                    Index_Acc = mean(Acc_SubPredict);%每一组参数对应的准确率
                                                    Index_Acc_ = mean(Acc_SubPredict_);
                                                    
                                                    if Index_Acc>Best_Acc
                                                        Best_Acc = Index_Acc;
                                                        Best_lambda1 = lambda1;
                                                        Best_lambda2 = lambda2;
                                                        Best_lambda3 = lambda3;
                                                        Best_lambda4 = lambda4;
                                                        Best_C1 = C1;
                                                        Best_C2 = C2;
                                                        Best_C3 = C3;
                                                        Best_C4 = C4;                       
                                                        Best_Kernel = Kernel;
                                                    end
                                                    
                                                    if Index_Acc_>Best_Acc_
                                                        Best_Acc_ = Index_Acc_;
                                                        Best_lambda1_ = lambda1;
                                                        Best_lambda2_ = lambda2;
                                                        Best_lambda3_ = lambda3;
                                                        Best_lambda4_ = lambda4;
                                                        Best_C1_ = C1;
                                                        Best_C2_ = C2;
                                                        Best_C3_ = C3;
                                                        Best_C4_ = C4;                       
                                                        Best_Kernel_ = Kernel;
                                                    end

                                                end  % gamma
                                  end

                              end    % lambda2
                        end
                    end    % lambda1
                    
                    Samples_A = Data_Train(Data_Train(:,end)==1, 1:end-1);
                    Labels_A = Data_Train(Data_Train(:,end)==1, end);
                    Samples_B = Data_Train(Data_Train(:,end)==-1, 1:end-1);
                    Labels_B = Data_Train(Data_Train(:,end)==-1, end);
                    
                    BestC_s.C1 = Best_C1;
                    BestC_s.C2 = Best_C2;
                    BestC_s.C3 = Best_C3;
                    BestC_s.C4 = Best_C4;
                    BestC_s.s1 = Fuzzy_MemberShip(Samples_A, Labels_A, Best_Kernel, Best_u);
                    BestC_s.s2 = Fuzzy_MemberShip(Samples_B, Labels_B, Best_Kernel, Best_u);
                    
                    BestC_s_.C1 = Best_C1_;
                    BestC_s_.C2 = Best_C2_;
                    BestC_s_.C3 = Best_C3_;
                    BestC_s_.C4 = Best_C4_;
                    
                    tic         
                    Outs_Train = Train_FTLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, Best_lambda1, Best_lambda2, BestC_s, Best_Kernel, QPPs_Solver);
                    t_Train(Times) = toc;
                    
                    tic         
                    Outs_Train_ = Train_TLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, Best_lambda1_, Best_lambda2_, BestC_s_, Best_Kernel_, QPPs_Solver);
                    t_Train_(Times) = toc;
                    
                   % Predict the data
                    Samples_Predict = Data_Predict(:, 1:end-1);
                    Labels_Predict  = Data_Predict (:, end);
                    Acc = Predict_FTLDM(Outs_Train, Samples_Predict,Labels_Predict, Samples_Train);
                    Acc_ = Predict_TLDM(Outs_Train_, Samples_Predict,Labels_Predict, Samples_Train);

                    Acc_Predict(Times) = Acc;
                    Acc_Predict_(Times) = Acc_;

                end
                
                
                %%%%%%%---------------------Save the statiatics---------------------%%%%%%%
                Name = Str_Name(10:end-4);
                %                 Loc_Nam = [Location, '\', Name, '.txt'];
                %                 f = fopen(Loc_Nam, 'wt');
                disp('FTBLDM');
                fprintf('%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train)), '.']);
                fprintf('%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(100*Acc_Predict)) '%.']);
%                 fprintf('%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(100*Acc_Predict)), '.']);
%                 fprintf('%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
%                 fprintf('%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);
                
                fprintf('%s\n', 'The Best_gamma is: ',num2str(Best_Kernel.gamma));
                fprintf('%s\n', 'The Best_lambda1 is: ',num2str(Best_lambda1));
                fprintf('%s\n','The Best_lambda2 is:',num2str(Best_lambda2));
                fprintf('%s\n', 'The Best_lambda3 is: ',num2str(Best_lambda3));
                fprintf('%s\n','The Best_lambda4 is:',num2str(Best_lambda4));

                fprintf('%s\n', 'The Best_C1 is: ',num2str(Best_C1));
                fprintf('%s\n', 'The Best_C2 is: ',num2str(Best_C2));
                fprintf('%s\n', 'The Best_C3 is: ',num2str(Best_C3));
                fprintf('%s\n','The Best_C4 is:',num2str(Best_C4));
                
                fprintf('%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train_)), '.']);
                fprintf('%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(100*Acc_Predict_)) '%.']);
%                 fprintf('%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(100*Acc_Predict)), '.']);
%                 fprintf('%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
%                 fprintf('%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);

                disp('TBLDM');
                fprintf('%s\n', 'The Best_gamma is: ',num2str(Best_Kernel_.gamma));
                fprintf('%s\n', 'The Best_lambda1 is: ',num2str(Best_lambda1_));
                fprintf('%s\n','The Best_lambda2 is:',num2str(Best_lambda2_));
                fprintf('%s\n', 'The Best_lambda3 is: ',num2str(Best_lambda3_));
                fprintf('%s\n','The Best_lambda4 is:',num2str(Best_lambda4_));

                fprintf('%s\n', 'The Best_C1 is: ',num2str(Best_C1_));
                fprintf('%s\n', 'The Best_C2 is: ',num2str(Best_C2_));
                fprintf('%s\n', 'The Best_C3 is: ',num2str(Best_C3_));
                fprintf('%s\n','The Best_C4 is:',num2str(Best_C4_));                
                
                temp = [mean(t_Train),mean(100*Acc_Predict),...
                   Best_Kernel.gamma, Best_lambda1,Best_lambda2,Best_lambda3,Best_lambda4,Best_C1,Best_C2,Best_C3,Best_C4];
                xlswrite('result.xlsx',temp,['B',num2str(iData+2),':L',num2str(iData+2)]);
                
                temp_ = [mean(t_Train_),mean(100*Acc_Predict_),...
                   Best_Kernel_.gamma, Best_lambda1_,Best_lambda2_,Best_la mbda3_,Best_lambda4_,Best_C1_,Best_C2_,Best_C3_,Best_C4_];
                xlswrite('result.xlsx',temp_,['N',num2str(iData+2),':X',num2str(iData+2)]);             
            end
       case 2   %%TLDM
            %% Some parameters
            % F_LDM Type
            FLDM_Type = 'F2_LDM';
            Kernel.Type = 'RBF';
            QPPs_Solver = 'QP_Matlab';
            gamma_Interval = 2.^(-6: 6);
            lambda1_Interval = 2.^(-6: 6);
            lambda2_Interval = 2.^(-6: 6);
%             C1_Interval = 1;
            C3_Interval = 2.^(-6: 6);

            
            Best_u = 0.1;
            
            
            %% Counts
            N_Times = 1;
            K_fold = 5;
            switch Kernel.Type
                case 'Linear'
                    Stop_Num = length(Str)*N_Times*length(lambda1_Interval)*length(lambda2_Interval)*length(C3_Interval) + 1;
                case 'RBF'
                    Stop_Num = 1*N_Times*length(lambda1_Interval)*length(lambda2_Interval)*length(C3_Interval)*length(gamma_Interval) + 1;
                otherwise
                    disp('  Wrong kernel function is provided.')
                    return
            end
            
            
            %% Train and predict the data
            for iData = 16:16
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
                    for ith_lambda1 = 1:length(lambda1_Interval)    % lambda1
                        lambda1 = lambda1_Interval(ith_lambda1);    % lambda1
                        lambda3 = lambda1;
                        for ith_lambda2 = 1:length(lambda2_Interval)
                          lambda2 = lambda2_Interval(ith_lambda2);
                          lambda4 = lambda2;

                          
%                                for ith_C1 = 1:length(C1_Interval)    % lambda3
%                                    C1 = C1_Interval(ith_C1);    % lambda3
%                                    C2 = C1;
                                  for ith_C3 = 1:length(C3_Interval) 
                                       C3 = C3_Interval(ith_C3);    % lambda3
                                       C4 = C3;
 
                                                for ith_gamma = 1:length(gamma_Interval)       %   gamma

                                                    Indices = crossvalind('Kfold', length(Labels_Train), K_fold);
                                                    Acc_SubPredict = zeros(1, 1);
                                                    for repeat = 1:1

                                                        I_SubTrain = ~(Indices==repeat);
                                                        Samples_SubTrain = Samples_Train(I_SubTrain, :);
                                                        Labels_SubTrain = Labels_Train(I_SubTrain, :);

                                                        %%%%%%-------Computes the average distance between instances-------%%%%%%
                                                        M_Sub = size(Samples_SubTrain, 1);
                                                        Index_Sub = combntns(1:M_Sub, 2); % Combination
                                                        delta_Sub = 0;
                                                        Num_Sub = size(Index_Sub, 1);
                                                        for i = 1:Num_Sub
                                                            delta_Sub = delta_Sub + norm(Samples_SubTrain(Index_Sub(i, 1), :)-Samples_SubTrain(Index_Sub(i, 2),:), 2)/Num_Sub;
                                                        end
                                                        %%%%%%-------Computes the average distance between instances-------%%%%%%
                                                        Kernel.gamma = delta_Sub*gamma_Interval(ith_gamma);  %   gamma

                                                        I_SubA = Labels_SubTrain == 1;
                                                        Samples_SubA = Samples_SubTrain(I_SubA,:);
                                                        Labels_SubA = Labels_SubTrain(I_SubA);

                                                        I_SubB = Labels_SubTrain == -1;
                                                        Samples_SubB = Samples_SubTrain(I_SubB,:);                        
                                                        Labels_SubB = Labels_SubTrain(I_SubB);     

%                                                         C_s.C1 = C1;
%                                                         C_s.C2 = C2;
                                                        C_s.s2 = Fuzzy_MemberShip(Samples_SubB, Labels_SubB, Kernel, Best_u);
                                                        C_s.C3 = C3;
                                                        C_s.C4 = C4;
                                                        C_s.s1 = Fuzzy_MemberShip(Samples_SubA, Labels_SubA, Kernel, Best_u);                                   


                                                        Outs_SubTrain = Train_TLDM(Samples_SubA, Labels_SubA, Samples_SubB,Labels_SubB, Samples_SubTrain, lambda1,lambda2 , C_s, Kernel, QPPs_Solver);

                                                        % Subpredict
                                                        I_SubPredict = ~ I_SubTrain;
                                                        Samples_SubPredict = Samples_Train(I_SubPredict, :);
                                                        Labels_SubPredict = Labels_Train(I_SubPredict, :);


                                                        SubAcc = Predict_TLDM(Outs_SubTrain, Samples_SubPredict,Labels_SubPredict, Samples_SubTrain);                                    
                                                        Acc_SubPredict(repeat) = SubAcc;

                                                        Stop_Num = Stop_Num - 1;
                                                        disp([num2str(Stop_Num), ' step(s) remaining.'])

                                                    end

                                                    Index_Acc = mean(Acc_SubPredict);
                                                    if Index_Acc>Best_Acc
                                                        Best_Acc = Index_Acc;
                                                        Best_lambda1 = lambda1;
                                                        Best_lambda2 = lambda2;
                                                        Best_lambda3 = lambda3;
                                                        Best_lambda4 = lambda4;
%                                                         Best_C1 = C1;
%                                                         Best_C2 = C2;
                                                        Best_C3 = C3;
                                                        Best_C4 = C4;                       
                                                        Best_Kernel = Kernel;
                                                    end


                                                end  % gamma
                                  end   %C3

%                                end    % C1  
                        end    % lambda2
                    end    % lambda1
                    
                    Samples_A = Data_Train(Data_Train(:,end)==1, 1:end-1);
                    Labels_A = Data_Train(Data_Train(:,end)==1, end);
                    Samples_B = Data_Train(Data_Train(:,end)==-1, 1:end-1);
                    Labels_B = Data_Train(Data_Train(:,end)==-1, end);
                    
%                     BestC_s.C1 = Best_C1;
%                     BestC_s.C2 = Best_C2;
                    BestC_s.C3 = Best_C3;
                    BestC_s.C4 = Best_C4;
                    BestC_s.s1 = Fuzzy_MemberShip(Samples_A, Labels_A, Best_Kernel, Best_u);

                    BestC_s.s2 = Fuzzy_MemberShip(Samples_B, Labels_B, Best_Kernel, Best_u);
                    tic
                    %                     Best_Cs.s = Fuzzy_MemberShip(Samples_Train, Labels_Train, Best_Kernel, Best_u);
%                     [centers_3,U] = fcm(Samples_Train,2);
%                     Best_Cs.s = (max(U))';
                    
                    Outs_Train = Train_TLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, Best_lambda1, Best_lambda2, BestC_s, Best_Kernel, QPPs_Solver);
                    t_Train(Times) = toc;
                    
                   % Predict the data
                    Samples_Predict = Data_Predict(:, 1:end-1);
                    Labels_Predict  = Data_Predict (:, end);
                     Acc = Predict_TLDM(Outs_Train, Samples_Predict,Labels_Predict, Samples_Train);

                    Acc_Predict(Times) = Acc;

                end
                
                
                %%%%%%%---------------------Save the statiatics---------------------%%%%%%%
                Name = Str_Name(10:end-4);
                %                 Loc_Nam = [Location, '\', Name, '.txt'];
                %                 f = fopen(Loc_Nam, 'wt');
                fprintf('%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train)), '.']);
                fprintf('%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(100*Acc_Predict)) '%.']);
%                 fprintf('%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(100*Acc_Predict)), '.']);
%                 fprintf('%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
%                 fprintf('%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);
                
                fprintf('%s\n', 'The Best_gamma is: ',num2str(Best_Kernel.gamma));
                fprintf('%s\n', 'The Best_lambda1 is: ',num2str(Best_lambda1));
                fprintf('%s\n','The Best_lambda2 is:',num2str(Best_lambda2));
                fprintf('%s\n', 'The Best_lambda3 is: ',num2str(Best_lambda3));
                fprintf('%s\n','The Best_lambda4  is:',num2str(Best_lambda4));
%                 fprintf('%f\n', Best_lambda1);
%                 fprintf('%s\n', 'The Best_lambda2 is: ');
%                 fprintf('%f\n', Best_lambda2);
%                 fprintf('%s\n', 'The Best_C1 is: ',num2str(Best_C1));
%                 fprintf('%s\n', 'The Best_C2 is: ',num2str(Best_C2));
                fprintf('%s\n', 'The Best_C3 is: ',num2str(Best_C3));
                fprintf('%s\n','The Best_C4 is:',num2str(Best_C4))
%                 fprintf('%f\n', Best_C);
%                 if  strcmp(Best_Kernel.Type, 'RBF')
%                     fprintf('%s\n', 'The Best_gamma is: ');
%                     fprintf('%f\n', Best_Kernel.gamma);
%                 end
%                 %                 fclose(f);
                temp = [mean(t_Train),mean(100*Acc_Predict),...
                   Best_Kernel.gamma, Best_lambda1,Best_lambda2,Best_lambda3,Best_lambda4,Best_C3,Best_C4];
                xlswrite('result.xlsx',temp,['AL',num2str(iData+2),':AT',num2str(iData+2)]);
            end            

        case 3   %%TBSVM
            %% Some parameters
            % F_LDM Type
            FLDM_Type = 'F2_LDM';
            Kernel.Type = 'RBF';
            QPPs_Solver = 'QP_Matlab';
            gamma_Interval = 2.^(-6:6);
            C1_Interval = 2.^(-6:6);
            C3_Interval = 2.^(-6:6);

            
            
            
            %% Counts
            N_Times = 1;
            K_fold = 5;
            switch Kernel.Type
                case 'Linear'
                    Stop_Num = length(Str)*N_Times*length(C1_Interval)*length(C3_Interval) + 1;
                case 'RBF'
                    Stop_Num = 1*N_Times*length(C1_Interval)*length(C3_Interval)*length(gamma_Interval) + 1;
                otherwise
                    disp('  Wrong kernel function is provided.')
                    return
            end
            
            
            %% Train and predict the data
            for iData = 16:16
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


                          
                               for ith_C1 = 1:length(C1_Interval)    % lambda3
                                   C1 = C1_Interval(ith_C1);    % lambda3
                                   C2 = C1;
                                   for ith_C3 = 1:length(C3_Interval)    % lambda3
                                       C3 = C3_Interval(ith_C3);    % lambda3
                                       C4 = C3;
 
                                                for ith_gamma = 1:length(gamma_Interval)       %   gamma

                                                    Indices = crossvalind('Kfold', length(Labels_Train), K_fold);
                                                    Acc_SubPredict = zeros(1, 1);
                                                    for repeat = 1:1

                                                        I_SubTrain = ~(Indices==repeat);
                                                        Samples_SubTrain = Samples_Train(I_SubTrain, :);
                                                        Labels_SubTrain = Labels_Train(I_SubTrain, :);

                                                        %%%%%%-------Computes the average distance between instances-------%%%%%%
                                                        M_Sub = size(Samples_SubTrain, 1);
                                                        Index_Sub = combntns(1:M_Sub, 2); % Combination
                                                        delta_Sub = 0;
                                                        Num_Sub = size(Index_Sub, 1);
                                                        for i = 1:Num_Sub
                                                            delta_Sub = delta_Sub + norm(Samples_SubTrain(Index_Sub(i, 1), :)-Samples_SubTrain(Index_Sub(i, 2),:), 2)/Num_Sub;
                                                        end
                                                        %%%%%%-------Computes the average distance between instances-------%%%%%%
                                                        Kernel.gamma = delta_Sub*gamma_Interval(ith_gamma);  %   gamma

%                                                         I_SubA = Labels_SubTrain == 1;
%                                                         Samples_SubA = Samples_SubTrain(I_SubA,:);
%                                                         Labels_SubA = Labels_SubTrain(I_SubA);
% 
%                                                         I_SubB = Labels_SubTrain == -1;
%                                                         Samples_SubB = Samples_SubTrain(I_SubB,:);                        
%                                                         Labels_SubB = Labels_SubTrain(I_SubB);     

 
                                                        Parameter.ker = 'rbf';
                                                        Parameter.CC = C3;
                                                        Parameter.CR = C1;
                                                        Parameter.p1 = Kernel.gamma;
                                                        Parameter.algorithm = 'QP';
                                                        Parameter.showplots = false;

%                                                         Outs_SubTrain = Train_TLDM(Samples_SubA, Labels_SubA, Samples_SubB,Labels_SubB, Samples_SubTrain, lambda1,lambda2 , C_s, Kernel, QPPs_Solver);

                                                        [tbsvm_Substruct] = tbsvmtrain(Samples_SubTrain,Labels_SubTrain,Parameter);
                                                        
                                                        % Subpredict
                                                        I_SubPredict = ~ I_SubTrain;
                                                        Samples_SubPredict = Samples_Train(I_SubPredict, :);
                                                        Labels_SubPredict = Labels_Train(I_SubPredict, :);


%                                                         SubAcc = Predict_FTLDM(Outs_SubTrain, Samples_SubPredict,Labels_SubPredict, Samples_SubTrain);           
                                                        [SubAcc]= tbsvmclass(tbsvm_Substruct,Samples_SubPredict,Labels_SubPredict);
                                                        
                                                        Acc_SubPredict(repeat) = SubAcc;

                                                        Stop_Num = Stop_Num - 1;
                                                        disp([num2str(Stop_Num), ' step(s) remaining.'])

                                                    end

                                                    Index_Acc = mean(Acc_SubPredict);
                                                    if Index_Acc>Best_Acc
                                                        Best_Acc = Index_Acc;
                                                        Best_C1 = C1;
                                                        Best_C2 = C2;
                                                        Best_C3 = C3;
                                                        Best_C4 = C4;                       
                                                        Best_Kernel = Kernel;
                                                    end


                                                end  % gamma
                                   end   %C3
                               
                               end     %C1

                    
                    Samples_A = Data_Train(Data_Train(:,end)==1, 1:end-1);
                    Labels_A = Data_Train(Data_Train(:,end)==1, end);
                    Samples_B = Data_Train(Data_Train(:,end)==-1, 1:end-1);
                    Labels_B = Data_Train(Data_Train(:,end)==-1, end);
                    
                    Parameter.ker = 'rbf';
                    Parameter.CC = Best_C3;
                    Parameter.CR = Best_C1;
                    Parameter.algorithm = 'QP';
                    Parameter.p1 = Best_Kernel.gamma;
                    tic
                    %                     Best_Cs.s = Fuzzy_MemberShip(Samples_Train, Labels_Train, Best_Kernel, Best_u);
%                     [centers_3,U] = fcm(Samples_Train,2);
%                     Best_Cs.s = (max(U))';
                    
%                     Outs_Train = Train_TLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, Best_lambda1, Best_lambda2, BestC_s, Best_Kernel, QPPs_Solver);
                    [tbsvm_struct] = tbsvmtrain(Samples_Train,Labels_Train,Parameter);
                    t_Train(Times) = toc;
                    
                   % Predict the data
                    Samples_Predict = Data_Predict(:, 1:end-1);
                    Labels_Predict  = Data_Predict (:, end);
%                     Acc = Predict_TLDM(Outs_Train, Samples_Predict,Labels_Predict, Samples_Train);
                   [Acc]= tbsvmclass(tbsvm_struct,Samples_Predict,Labels_Predict);
                    Acc_Predict(Times) = Acc;

                end
                
                
                %%%%%%%---------------------Save the statiatics---------------------%%%%%%%
                Name = Str_Name(10:end-4);
                %                 Loc_Nam = [Location, '\', Name, '.txt'];
                %                 f = fopen(Loc_Nam, 'wt');
                fprintf('%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train)), '.']);
                fprintf('%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(Acc_Predict)) '%.']);
%                 fprintf('%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(Acc_Predict)), '.']);
%                 fprintf('%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
%                 fprintf('%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);
                
                fprintf('%s\n', 'The Best_gamma is: ',num2str(Best_Kernel.gamma));
%                 fprintf('%f\n', Best_lambda1);
%                 fprintf('%s\n', 'The Best_lambda2 is: ');
%                 fprintf('%f\n', Best_lambda2);
                fprintf('%s\n', 'The Best_C1 is: ',num2str(Best_C1));
                fprintf('%s\n', 'The Best_C2 is: ',num2str(Best_C2));
                fprintf('%s\n', 'The Best_C3 is: ',num2str(Best_C3));
                fprintf('%s\n','The Best_C4 is:',num2str(Best_C4))
%                 fprintf('%f\n', Best_C);
%                 if  strcmp(Best_Kernel.Type, 'RBF')
%                     fprintf('%s\n', 'The Best_gamma is: ');
%                     fprintf('%f\n', Best_Kernel.gamma);
%                 end
%                 %                 fclose(f);
                temp = [mean(t_Train),mean(Acc_Predict),...
                   Best_Kernel.gamma,Best_C1,Best_C2,Best_C3,Best_C4];
                xlswrite('result.xlsx',temp,['AA',num2str(iData+21),':AG',num2str(iData+21)]);
            end    
    
       case 4   %%FTSVM and TSVM
            %% Some parameters
            % F_LDM Type
            FLDM_Type = 'F2_LDM';
            Kernel.Type = 'RBF';
            QPPs_Solver = 'QP_Matlab';
            gamma_Interval = 2.^(-6: 6);
            C1_Interval = 2.^(-6: 6);

            
            Best_u = 0.1;
            
            
            %% Counts
            N_Times = 1;
            K_fold = 5;
            switch Kernel.Type
                case 'Linear'
                    Stop_Num = 7*N_Times*length(lambda1_Interval)*length(lambda2_Interval)*length(C1_Interval)*length(C3_Interval) + 1;
                case 'RBF'
                    Stop_Num = 18*N_Times*length(C1_Interval)*length(gamma_Interval) + 1;
                otherwise
                    disp('  Wrong kernel function is provided.')
                    return
            end
            
            
            %% Train and predict the data
            for iData = 1:18
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
                t_Train_ = zeros(N_Times, 1);
                Acc_Predict = zeros(N_Times, 1);
                Acc_Predict_ = zeros(N_Times, 1);
                MarginMEAN_Train = zeros(N_Times, 1);
                MarginSTD_Train = zeros(N_Times, 1);
                Accuracy = zeros();
                for Times = 1: N_Times
                    
                        [Data_Train, Data_Predict] = Data_Rate(Data_Original, TrainRate);   % Chose 3

                        Samples_Train = Data_Train(:, 1:end-1);
                        Labels_Train = Data_Train(:, end);

                        Best_Acc = 0;
                        Best_lambda1 = 0;
                        Best_lambda2 = 0;
                        Best_lambda3 = 0;
                        Best_lambda4 = 0;
                        Best_C1 = 0;
                        Best_C2 = 0;
                        Best_C3 = 0;
                        Best_C4 = 0;   

                        Best_Acc_ = 0;
                        Best_lambda1_ = 0;
                        Best_lambda2_ = 0;
                        Best_lambda3_ = 0;
                        Best_lambda4_ = 0;
                        Best_C1_ = 0;
                        Best_C2_ = 0;
                        Best_C3_ = 0;
                        Best_C4_ = 0;                      





                           for ith_C1 = 1:length(C1_Interval)    % lambda3
                               C1 = C1_Interval(ith_C1);    % lambda3
                               C2 = C1;


                                    for ith_gamma = 1:length(gamma_Interval)       %   gamma

                                        Indices = crossvalind('Kfold', length(Labels_Train), K_fold);%随机分成K_fold份，并打上标签
                                        Acc_SubPredict = zeros(1, 1);
                                        Acc_SubPredict_ = zeros(1, 1);
                                        for repeat = 1:1

                                            I_SubTrain = ~(Indices==repeat);%(Kfold-1)/Kfold 拿来训练
                                            Samples_SubTrain = Samples_Train(I_SubTrain, :);
                                            Labels_SubTrain = Labels_Train(I_SubTrain, :);

                                            %%%%%%-------Computes the average distance between instances-------%%%%%%
                                            M_Sub = size(Samples_SubTrain, 1);
                                            Index_Sub = combntns(1:M_Sub, 2); % Combination
                                            delta_Sub = 0;
                                            Num_Sub = size(Index_Sub, 1);
                                            for i = 1:Num_Sub
                                                delta_Sub = delta_Sub + norm(Samples_SubTrain(Index_Sub(i, 1), :)-Samples_SubTrain(Index_Sub(i, 2),:), 2)/Num_Sub;
                                            end
                                            %%%%%%-------Computes the average distance between instances-------%%%%%%
                                            Kernel.gamma = delta_Sub*gamma_Interval(ith_gamma);  %   gamma

                                            I_SubA = Labels_SubTrain == 1;
                                            Samples_SubA = Samples_SubTrain(I_SubA,:);
                                            Labels_SubA = Labels_SubTrain(I_SubA);

                                            I_SubB = Labels_SubTrain == -1;
                                            Samples_SubB = Samples_SubTrain(I_SubB,:);                        
                                            Labels_SubB = Labels_SubTrain(I_SubB);     

                                            C_s.C1 = C1;
                                            C_s.C2 = C2;                                        
                                            C_s.s1 = Fuzzy_MemberShip(Samples_SubA, Labels_SubA, Kernel, Best_u);   
                                            C_s.s2 = Fuzzy_MemberShip(Samples_SubB, Labels_SubB, Kernel, Best_u);



                                            Outs_SubTrain = Train_FTSVM(Samples_SubA, Labels_SubA, Samples_SubB,Labels_SubB, Samples_SubTrain, C_s, Kernel, QPPs_Solver);
                                            Outs_SubTrain_ = Train_TSVM(Samples_SubA, Labels_SubA, Samples_SubB,Labels_SubB, Samples_SubTrain, C_s, Kernel, QPPs_Solver);

                                            % Subpredict
                                            I_SubPredict = ~ I_SubTrain;% 1/Kfold 拿来预测
                                            Samples_SubPredict = Samples_Train(I_SubPredict, :);
                                            Labels_SubPredict = Labels_Train(I_SubPredict, :);


                                            SubAcc = Predict_FTSVM(Outs_SubTrain, Samples_SubPredict,Labels_SubPredict, Samples_SubTrain);   
                                            SubAcc_ = Predict_TSVM(Outs_SubTrain, Samples_SubPredict,Labels_SubPredict, Samples_SubTrain);

                                            Acc_SubPredict(repeat) = SubAcc;
                                            Acc_SubPredict_(repeat) = SubAcc_;

                                            Stop_Num = Stop_Num - 1;

                                            disp([num2str(Stop_Num), ' step(s) remaining.'])

                                        end

                                        Index_Acc = mean(Acc_SubPredict);%每一组参数对应的准确率
                                        Index_Acc_ = mean(Acc_SubPredict_);

                                        if Index_Acc>Best_Acc
                                            Best_Acc = Index_Acc;
                                            Best_C1 = C1;
                                            Best_C2 = C2;                      
                                            Best_Kernel = Kernel;
                                        end

                                        if Index_Acc_>Best_Acc_
                                            Best_Acc_ = Index_Acc_;
                                            Best_C1_ = C1;
                                            Best_C2_ = C2;                     
                                            Best_Kernel_ = Kernel;
                                        end

                                    end
                           end

 


                        Samples_A = Data_Train(Data_Train(:,end)==1, 1:end-1);
                        Labels_A = Data_Train(Data_Train(:,end)==1, end);
                        Samples_B = Data_Train(Data_Train(:,end)==-1, 1:end-1);
                        Labels_B = Data_Train(Data_Train(:,end)==-1, end);

                        BestC_s.C1 = Best_C1;
                        BestC_s.C2 = Best_C2;

                        BestC_s.s1 = Fuzzy_MemberShip(Samples_A, Labels_A, Best_Kernel, Best_u);
                        BestC_s.s2 = Fuzzy_MemberShip(Samples_B, Labels_B, Best_Kernel, Best_u);

                        BestC_s_.C1 = Best_C1_;
                        BestC_s_.C2 = Best_C2_;

                        tic         
                        Outs_Train = Train_FTSVM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, BestC_s, Best_Kernel, QPPs_Solver);
                        t_Train(Times) = toc;

                        tic         
                        Outs_Train_ = Train_TSVM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train,BestC_s_, Best_Kernel_, QPPs_Solver);
                        t_Train_(Times) = toc;

                       % Predict the data
                        Samples_Predict = Data_Predict(:, 1:end-1);
                        Labels_Predict  = Data_Predict (:, end);
                        Acc = Predict_FTSVM(Outs_Train, Samples_Predict,Labels_Predict, Samples_Train);
                        Acc_ = Predict_TSVM(Outs_Train_, Samples_Predict,Labels_Predict, Samples_Train);

                        Acc_Predict(Times) = Acc;
                        Acc_Predict_(Times) = Acc_;

                    end
                
                
                %%%%%%%---------------------Save the statiatics---------------------%%%%%%%
                Name = Str_Name(10:end-4);
                %                 Loc_Nam = [Location, '\', Name, '.txt'];
                %                 f = fopen(Loc_Nam, 'wt');
                disp('FTSVM');
                fprintf('%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train)), '.']);
                fprintf('%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(100*Acc_Predict)) '%.']);
%                 fprintf('%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(100*Acc_Predict)), '.']);
%                 fprintf('%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
%                 fprintf('%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);
                
                fprintf('%s\n', 'The Best_gamma is: ',num2str(Best_Kernel.gamma));

                fprintf('%s\n', 'The Best_C1 is: ',num2str(Best_C1));
                fprintf('%s\n', 'The Best_C2 is: ',num2str(Best_C2));
                
                fprintf('%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train_)), '.']);
                fprintf('%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(100*Acc_Predict_)) '%.']);
%                 fprintf('%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(100*Acc_Predict)), '.']);
%                 fprintf('%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
%                 fprintf('%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);

                disp('TSVM');
                fprintf('%s\n', 'The Best_gamma is: ',num2str(Best_Kernel_.gamma));

                fprintf('%s\n', 'The Best_C1 is: ',num2str(Best_C1_));
                fprintf('%s\n', 'The Best_C2 is: ',num2str(Best_C2_));              
                
                temp = [mean(t_Train),mean(100*Acc_Predict),...
                   Best_Kernel.gamma,Best_C1,Best_C2];
                xlswrite('result.xlsx',temp,['Z',num2str(iData+2),':AD',num2str(iData+2)]);
                
                temp_ = [mean(t_Train_),mean(100*Acc_Predict_),...
                   Best_Kernel_.gamma,Best_C1_,Best_C2_];
                xlswrite('result.xlsx',temp_,['AF',num2str(iData+2),':AJ',num2str(iData+2)]);  

            end
            
    end
            
end

