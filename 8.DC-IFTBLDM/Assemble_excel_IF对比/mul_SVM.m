%% Initilizing the enviroment 
   clear 
   close 
   clc
   rng('default') 
   
   
%% Load and prepare the data
% %     Setting1
    load ('Data_mat\batchs1.mat');
    batchS = batchs{1}(:,4:2:end);
    batchS=batchS./repmat(sqrt(sum(batchS.^2,1)),size(batchS,1),1);%标准化
    batchS_label=batchs{1}(:,1);
    
    k = max(batchS_label(:,1));%k分类任务
    lambda = [0,0.008,0.9,1000,0.03,0.06,8,0.01,0.001,0.08];
    d = [0,30,40,39,12,12,17,8,11,10];

    
  % Normalization
  for i = 2:10
      
%       Setting2      
%     batchS = batchs{i-1}(:,4:2:end);
%     batchS=batchS./repmat(sqrt(sum(batchS.^2,1)),size(batchS,1),1);%标准化
%     batchS_label=batchs{i-1}(:,1);
%     
%     k = max(batchS_label(:,1));%k分类任务
%     lambda = [0,0.008,1,0.1,0.001,0.01,10000,0.1,0.09,0.1];
%     d = [0,30,84,61,8,110,109,15,88,55];
%     
      
    batchT = batchs{i}(:,4:2:end);
    batchT=batchT./repmat(sqrt(sum(batchT.^2,1)),size(batchT,1),1);%标准化
    batchT_label=batchs{i}(:,1);
    
    
    
    MbatchS=mean(batchS);
    MbatchT=mean(batchT);
    Ms=MbatchS';
    Mt=MbatchT';
    
    
    neweigenvector=DRCA(Ms,Mt,batchS',batchT',lambda(i));
    P=neweigenvector(:,1:d(i));
    batchS_P=batchS*P;
    batchT_P=batchT*P;
    
    batchS_P = [batchS_P,batchS_label];
    batchT_P = [batchT_P,batchT_label];
 
    
%% Some parameters
   Kernel.Type = 'Linear';
   QPPs_Solver = 'qp';  % 'qp', 'QP_Matlab'
   C_Interval = [1 10 50 100 500];
   Best_C = 2*max(C_Interval);
   if strcmp(Kernel.Type, 'RBF')
       gamma_Interval = 2.^(-4:4);
       Value_Contour = 1;
       Str_Legend = 'RBF-kernel SVM';
   elseif strcmp(Kernel.Type, 'Linear')
       Value_Contour = 1;
       Str_Legend = 'Linear SVM';
   else
       disp('Wrong parameters are provided.')
       return
   end

 
  %% Counts
     N_Times = 10;
     K_fold = 5;
     switch Kernel.Type
         case 'Linear'
             Stop_Num = N_Times*length(C_Interval)*K_fold + 1;
         case 'RBF'
             Stop_Num = N_Times*length(C_Interval)*length(gamma_Interval)*K_fold + 1;
         otherwise
             disp('  Wrong kernel function is provided.')
             return
     end
     
     
%% Training prodecure 训练
    TrainRate = 0.8;       % The scale of the train set 训练规模设置
    t_Train = zeros(N_Times, 1);
    Acc_Predict = zeros(N_Times, 1);
    Acc_Leader = 0;
    MarginMEAN_Train = zeros(N_Times, 1);
    MarginSTD_Train = zeros(N_Times, 1);
    for i1 = 1:k
       for j1 = 1:k
           if i1 < j1
                    for Times = 1: N_Times  %寻找最优        

                        Data_Train = batchS_P;
                        Data_Predict = batchT_P;
                        voting_mat = zeros(length(Data_Predict(:,end)),k);

                        Best_Acc = 0;
                        Acc_SubPredict = zeros(K_fold, 1);

                        class_i = Data_Train(find(Data_Train(:,end) == i1),:);
                        class_j = Data_Train(find(Data_Train(:,end) == j1),:);
                        Labels_Train = [ones(size(class_i,1),1);-ones(size(class_j,1),1)];
                        Ord_Samples_Train = [class_i;class_j];
                        Ord_Samples_Train(:,end) = Labels_Train;
                        %数据重新打乱,模拟真实数据
                        Disord_Samples_Train = Ord_Samples_Train(randperm(size(Ord_Samples_Train, 1)), :);
                        Samples_Train = Disord_Samples_Train(:,1:end-1);
                        Labels_Train =Disord_Samples_Train(:,end);

                        for ith_C = 1:length(C_Interval)    %  C
                            C = C_Interval(ith_C);          %  C

                %             for ith_gamma = 1:length(gamma_Interval)       %   gamma

                              % CV parameters
                                Indices = crossvalind('Kfold', length(Labels_Train), K_fold);
                                for repeat = 1:K_fold
                                  % Subtrain
                                    I_SubTrain = ~(Indices==repeat); %%%**************************************************************************************？？？
                                    Samples_SubTrain = Samples_Train(I_SubTrain, :);
                                    Labels_SubTrain = Labels_Train(I_SubTrain, :);

                %                   %%%%%%-------Computes the average distance between instances-------%%%%%%
                %                     M_Sub = size(Samples_SubTrain, 1);
                %                     Index_Sub = combntns(1:M_Sub, 2); % Combination
                %                     delta_Sub = 0;
                %                     Num_Sub = size(Index_Sub, 1);
                %                     for i = 1:Num_Sub
                %                         delta_Sub = delta_Sub + norm(Samples_SubTrain(Index_Sub(i, 1), :)-Samples_SubTrain(Index_Sub(i, 2),:), 2)/Num_Sub;
                %                     end
                %                   %%%%%%-------Computes the average distance between instances-------%%%%%%
                %                     Kernel.gamma = delta_Sub*gamma_Interval(ith_gamma);  %   gamma

                                    Outs_SubTrain = Train_SVM(Samples_SubTrain, Labels_SubTrain, C*abs(Labels_SubTrain), Kernel, QPPs_Solver);

                                  % Subpredict
                                    I_SubPredict = ~ I_SubTrain;
                                    Samples_SubPredict = Samples_Train(I_SubPredict, :); 
                                    Labels_SubPredict = Labels_Train(I_SubPredict, :);
                                    SubAcc = Predict_SVM(Outs_SubTrain, Samples_SubPredict, Labels_SubPredict);
                                    Acc_SubPredict(repeat) = SubAcc;

                                    Stop_Num = Stop_Num - 1;
                                    %%%disp([num2str(Stop_Num), ' step(s) remaining.'])
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

                        if Acc>Acc_Leader
                            Acc_Leader = Acc;
%                             C_Leader = Best_C;
%                             Kernel_Leader = Best_Kernel;
%                             Margin_Leader = Margin; 
%                             Samples_Leader = Samples_Train;
%                             Labels_Leader = Labels_Train;
%                             Supporters_Leader = Data_Supporters;
%                             Outs_Leader = Outs_Predict;
                            Label_Decision_Leader = Label_Decision;
                        end

                    end
                    for i2 = 1:length(Samples_Predict)
                        if Label_Decision_Leader(i2) == 1
                           voting_mat(i2,i1) = voting_mat(i2,i1)+1;
                        else
                           voting_mat(i2,j1) = voting_mat(i2,j1)+1;
                        end
                    end
           end
       end
     end
                     % Voting result
    pred_label = zeros(length(Samples_Predict),1);
    for i3 = 1:length(Samples_Predict)
        [max_val,max_ind] =  max(voting_mat(i3,:));
        pred_label(i3) = max_ind;
    end
    mul_Acc = (length(find(Labels_Predict == pred_label)))/length(Labels_Predict); 
%   disp(mul_Acc); 
    s = "C" + i;
    xlswrite(".\results\Setting1.xlsx",mul_Acc, 1, s);                    
 end
                    

disp('END!')
    

   