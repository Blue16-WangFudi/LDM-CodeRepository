%% Initilizing the enviroment
clear all
close all
clc

%% Data preparation



%%  Choose the Classifier
% case 1:FTBLDM and TBLDM; 2:TLDM; 3:TBSVM; 4:FTSVM and TSVM;
%% Some parameters
% F_LDM Type
FLDM_Type = 'F2_LDM';
Kernel.Type = 'RBF';
QPPs_Solver = 'QP_Matlab';
% gamma_Interval = 2.^(-5:2);
% lambda1_Interval = 2.^(-8:4);
% lambda2_Interval = 2.^(-8:4);
% C1_Interval = 2.^(-8:4);
% C3_Interval = 2.^(-8:4);

gamma_Interval = 2.^(-5:-4);
lambda1_Interval = 2.^(-8:-7);
lambda2_Interval = 2.^(-8:-7);
C1_Interval = 2.^(-8:-7);
C3_Interval = 2.^(-8:-7);


Best_u = 0.4;


%% Counts
N_Times = 1;
switch Kernel.Type
    case 'Linear'
        Stop_Num = 7*N_Times*length(lambda1_Interval)*length(lambda2_Interval)*length(C1_Interval)*length(C3_Interval) + 1;
    case 'RBF'
        Stop_Num = 1*N_Times*length(lambda1_Interval)*length(lambda2_Interval)*length(C1_Interval)*length(C3_Interval)*length(gamma_Interval) + 1;
    otherwise
        disp('  Wrong kernel function is provided.')
        return
end

    mul=[1,-1.5]; % 均值
    S1=[3 0;0 3]; % 协方差
    data1=mvnrnd(mul, S1, 300); % 产生高斯分布数据
    data1(:,3)=1;
    % 第二组数据
    mu2=[-1,2];
    S2=[2 0;0 3];
    data2=mvnrnd(mu2,S2,300);
    data2(:,3)=-1;
    % noises of p
    mm1=70;
    %         mu3=[0,0.3];
    %         S3=[0.3 0;0 0.3];
    mu3=[-1,-0.6];
    S3=[0.5 0;0 0.7];
    data3=mvnrnd(mu3,S3,mm1);
    data3(:,3)=-1;
    % noises of n
    mm2=mm1;
    %         mu4=[0,-0.3];
    %         S4=[0.3 0.1;0.1 0.3];
    mu4=[1,0.5];
    S4=[0.5 0;0 0.7];
    data4=mvnrnd(mu4,S4,mm2);
    data4(:,3)=1;
    data_noise = [data3;data4];
    data_noise = data_noise(randperm(size(data_noise,1)),:);  
    plot(data1(1:300,1),data1(1:300,2),"o",data2(1:300,1),data2(1:300,2),"x" ,data_noise(1:70,1),data_noise(1:70,2),"+",data_noise(70:140,1),data_noise(70:140,2),"*");
    axis([-5 5,-5,5])
%    







 


    

      
    for type=1:1
        switch type
           case 1   %%DC-IFTBLDM and TBLDM
            %% Train and predict the data
               for i1 = 1:40
                    for j1 = 1:40
                        if i1 < j1
                            load('Data_mat\orl_train.mat');
                            load('Data_mat\orl_predict.mat');
                            Data_Train = orl_train;
                            Data_Predict = orl_predict;
                            voting_mat = zeros(length(Data_Predict(:,end)),40); 

                            class_i = Data_Train(find(Data_Train(:,end) == i1),:);
                            class_j = Data_Train(find(Data_Train(:,end) == j1),:);

                            Ctrain = [class_i;class_j];
                            dtrain = [ones(size(class_i,1),1);-ones(size(class_j,1),1)];    

                            Ctest = Data_Predict(:, 1:end-1);
                            dtest = Data_Predict(:, end);

                            Ctrain = svdatanorm(Ctrain,'svpline');
                            Ctest = svdatanorm(Ctest,'svpline');



                            %%%%%%%-----------------Training the best parameters-----------------%%%%%%%
                            TrainRate = 0.5;       % The scale of the tuning set
                            t_Train = zeros(N_Times, 1);
            %                 t_Train_ = zeros(N_Times, 1);
                            Acc_Predict = zeros(N_Times, 1);
            %                 Acc_Predict_ = zeros(N_Times, 1);

                            Accuracy = zeros();
                            for Times = 1: N_Times


                                Samples_Train = Ctrain;
                                Samples_Predict = Ctest;
                                Labels_Train = dtrain;                    
                                Labels_Predict = dtest;                    


                                Best_lambda1 = 0;
                                Best_lambda2 = 0;
                                Best_lambda3 = 0;
                                Best_lambda4 = 0;
                                Best_C1 = 0;
                                Best_C2 = 0;
                                Best_C3 = 0;
                                Best_C4 = 0;   


            %                     Best_lambda1_ = 0;
            %                     Best_lambda2_ = 0;
            %                     Best_lambda3_ = 0;
            %                     Best_lambda4_ = 0;
            %                     Best_C1_ = 0;
            %                     Best_C2_ = 0;
            %                     Best_C3_ = 0;
            %                     Best_C4_ = 0;  


                                C3_ACC = ones(1,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                                C3_lambda1 = ones(length(lambda1_Interval),1);
                                C3_lambda2 = ones(length(lambda2_Interval),1);
                                C3_C1 = ones(length(C1_Interval),1);
                                C3_C3 = ones(length(C3_Interval),1);
                                Gam = ones(length(gamma_Interval),1);

            %                     C3_ACC_ = ones(1,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
            %                     C3_lambda1_ = ones(length(lambda1_Interval),1);
            %                     C3_lambda2_ = ones(length(lambda2_Interval),1);
            %                     C3_C1_ = ones(length(C1_Interval),1);
            %                     C3_C3_ = ones(length(C3_Interval),1);
            %                     Gam_ = ones(length(gamma_Interval),1);


                                for ith_C3 = 1:length(C3_Interval) 
                                        C3 = C3_Interval(ith_C3);    % lambda3
                                        C4 = C3;

                                        Best_Acc = 0;%必须要清0，设为0后，其余参数就不需要再重置了。
                                        Best_Acc_ = 0;

                                        for ith_lambda1 = 1:length(lambda1_Interval)    % lambda1
                                            lambda1 = lambda1_Interval(ith_lambda1);    % lambda1
                                            lambda3 = lambda1;
                                            for ith_lambda2 = 1:length(lambda2_Interval)
                                              lambda2 = lambda2_Interval(ith_lambda2);
                                              lambda4 = lambda2;


                                                   for ith_C1 = 1:length(C1_Interval)    % lambda3
                                                       C1 = C1_Interval(ith_C1);    % lambda3
                                                       C2 = C1;


                                                                    for ith_gamma = 1:length(gamma_Interval)       %   gamma

                                                                        Acc_SubPredict = zeros(1, 1);
            %                                                             Acc_SubPredict_ = zeros(1, 1);


                                                                        %%%%%%-------Computes the average distance between instances-------%%%%%%
                    %                                                     M_Sub = size(Samples_Train, 1);
                    %                                                     Index_Sub = combntns(1:M_Sub, 2); % Combination
                    %                                                     delta_Sub = 0;
                    %                                                     Num_Sub = size(Index_Sub, 1);
                    %                                                     for i = 1:Num_Sub
                    %                                                         delta_Sub = delta_Sub + norm(Samples_Train(Index_Sub(i, 1), :)-Samples_Train(Index_Sub(i, 2),:), 2)/Num_Sub;
                    %                                                     end
                    %                                                     %%%%%%-------Computes the average distance between instances-------%%%%%%
                                                                        Kernel.gamma = gamma_Interval(ith_gamma);  %   gamma

                                                                        I_A = Labels_Train == 1;
                                                                        Samples_A = Samples_Train(I_A,:);
                                                                        Labels_A = Labels_Train(I_A);

                                                                        I_B = Labels_Train == -1;
                                                                        Samples_B = Samples_Train(I_B,:);                        
                                                                        Labels_B = Labels_Train(I_B);    



            %                                                             membership(Samples_Train', Labels_Train')
                                                                        s = IFuzzy_MemberShip(Samples_Train, Labels_Train, Kernel, Best_u); 
                                                                        C_s.C1 = C1;
                                                                        C_s.C2 = C2;
                                                                        C_s.s1 = s.s1;
                                                                        C_s.s2 = s.s2;
                                                                        C_s.C3 = C3;
                                                                        C_s.C4 = C4;



                                                                        Outs_Train = Train_FTBLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, lambda1,lambda2 , C_s, Kernel, QPPs_Solver);
            %                                                             Outs_Train_ = Train_TBLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, lambda1,lambda2 , C_s, Kernel, QPPs_Solver);
                                                                        [Acc,Label_Decision] = Predict_FTBLDM(Outs_Train, Samples_Predict,SubLabels_Predict, Samples_Train);   
            % %                                                             Acc_ = Predict_TBLDM(Outs_Train_, Samples_Predict,Labels_Predict, Samples_Train);


                                                                       Stop_Num = Stop_Num - 1;

                                                                        disp([num2str(Stop_Num), ' step(s) remaining.'])


                                                                        if Acc>Best_Acc
                                                                            Best_Acc = Acc;
                                                                            Best_lambda1 = lambda1;
                                                                            Best_lambda2 = lambda2; 
                                                                            Best_C1 = C1;
                                                                            Best_C3 = C3;                      
                                                                            Best_Kernel = Kernel;
                                                                            Label_Decision_Leader = Label_Decision;
                                                                        end
            %                                                             if Acc_>Best_Acc_
            %                                                                 Best_Acc_ = Acc_;
            %                                                                 Best_lambda1_ = lambda1;
            %                                                                 Best_lambda2_ = lambda2;
            %                                                                 Best_C1_ = C1;
            %                                                                 Best_C3_ = C3;                      
            %                                                                 Best_Kernel_ = Kernel;
            %                                                             end


                                                                    end  % gamma


                                                  end    % lambda2
                                            end
                                        end    % lambda1


                                        C3_ACC(ith_C3) = Best_Acc;
                                        C3_lambda1(ith_C3) = Best_lambda1;
                                        C3_lambda2(ith_C3) = Best_lambda2;
                                        C3_C1(ith_C3) = Best_C1;
                                        C3_C3(ith_C3) = Best_C3;
                                        Gam(ith_C3) = Best_Kernel.gamma;

            %                             C3_ACC_(ith_C3) = Best_Acc_;
            %                             C3_lambda1_(ith_C3) = Best_lambda1_;
            %                             C3_lambda2_(ith_C3) = Best_lambda2_;
            %                             C3_C1_(ith_C3) = Best_C1_;
            %                             C3_C3_(ith_C3) = Best_C3_;
            %                             Gam_(ith_C3) = Best_Kernel_.gamma;




                                end

                                [Best_Acc,i]=max(C3_ACC);
            %                     [Best_Acc_,j]=max(C3_ACC_);

                                BestC_s.C1 = C3_C1(i);
                                BestC_s.C2 = C3_C1(i);
                                BestC_s.C3 = C3_C3(i);
                                BestC_s.C4 = C3_C3(i);
                                Best_lambda1 = C3_lambda1(i);
                                Best_lambda2 = C3_lambda2(i);
                                Best_Kernel.gamma = Gam(i);
                                Bests = IFuzzy_MemberShip(Samples_Train, Labels_Train, Kernel, Best_u); 
                                BestC_s.s1 = Bests.s1;
                                BestC_s.s2 = Bests.s2;

            %                     BestC_s_.C1 = C3_C1_(j);
            %                     BestC_s_.C2 = C3_C1_(j);
            %                     BestC_s_.C3 = C3_C3_(j);
            %                     BestC_s_.C4 = C3_C3_(j);
            %                     Best_lambda1_ = C3_lambda1_(j);
            %                     Best_lambda2_ = C3_lambda2_(j);
            %                     Best_Kernel_.gamma = Gam_(j);


                                tic         
                                Outs_Train = Train_FTBLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, Best_lambda1, Best_lambda2, BestC_s, Best_Kernel, QPPs_Solver);
                                t_Train(Times) = toc;
            %                     tic         
            %                     Outs_Train_ = Train_TBLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, Best_lambda1_, Best_lambda2_, BestC_s_, Best_Kernel_, QPPs_Solver);
            %                     t_Train_(Times) = toc;


                               % Predict the data
                                Acc_Predict(Times) = Best_Acc;
            %                     Acc_Predict_(Times) = Best_Acc_;



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

            pred_label = zeros(length(Samples_Predict),1);
            for i3 = 1:length(Samples_Predict)
                [max_val,max_ind] =  max(voting_mat(i3,:));
                pred_label(i3) = max_ind;
            end
            mul_Acc = (length(find(Labels_Predict == pred_label)))/length(Labels_Predict); 


            temp = [mean(t_Train),mean(100*mul_Acc),...
               Best_Kernel.gamma, Best_lambda1,Best_lambda2,BestC_s.C1,BestC_s.C3];
            xlswrite('result.xlsx',temp,['C',num2str(iData+2),':I',num2str(iData+2)]);
            xlswrite('result.xlsx',100*C3_ACC,['C',num2str(iData+33),':O',num2str(iData+33)]);


%                 temp_ = [mean(t_Train_),mean(100*Acc_Predict_),...
%                    Best_Kernel_.gamma, Best_lambda1_,Best_lambda2_,BestC_s_.C1,BestC_s_.C3];
%                 xlswrite('result.xlsx',temp_,['K',num2str(iData+2),':Q',num2str(iData+2)]); 
%                 xlswrite('result.xlsx',C3_ACC_,['C',num2str(iData+52),':O',num2str(iData+52)]);


            case 2  %%TBSVM and IFTSVM and CDFSVM
            %% Train and predict the data
            for i1 = 1:40
                    for j1 = 1:40
                        if i1 < j1
                            load('Data_mat\orl_train.mat');
                            load('Data_mat\orl_predict.mat');
                            Data_Train = orl_train;
                            Data_Predict = orl_predict;
                            voting_mat = zeros(length(Data_Predict(:,end)),40);

                            class_i = Data_Train(find(Data_Train(:,end) == i1),:);
                            class_j = Data_Train(find(Data_Train(:,end) == j1),:);

                            Ctrain = [class_i;class_j];
                            dtrain = [ones(size(class_i,1),1);-ones(size(class_j,1),1)];                
                            Ctest = Data_Predict(:, 1:end-1);
                            dtest = Data_Predict(:, end);

                            Ctrain = svdatanorm(Ctrain,'svpline');
                            Cteat = svdatanorm(Cteat,'svpline');

                                %%%%%%%-----------------Training the best parameters-----------------%%%%%%%
                                TrainRate = 0.5;       % The scale of the tuning set
                                t_Train = zeros(N_Times, 1);
                                t_Train_ = zeros(N_Times, 1);
                                t_Train_1 = zeros(N_Times, 1);

                                Acc_Predict = zeros(N_Times, 1);
                                Acc_Predict_ = zeros(N_Times, 1);
                                Acc_Predict_1 = zeros(N_Times, 1);

                                MarginMEAN_Train = zeros(N_Times, 1);
                                MarginSTD_Train = zeros(N_Times, 1);
                                Accuracy = zeros();
                                for Times = 1: N_Times

                %                     Ctrain1=awgn(Ctrain,0.1); % 高斯噪声
                %                     Ctest1=awgn(Ctest,0.1); % 高斯噪声
                %                     Ctrain= svdatanorm(Ctrain,'svpline');
                %                     Ctest= svdatanorm(Ctest,'scpline');
                                    Samples_Train = Ctrain;
                                    Labels_Train = dtrain;
                                    Samples_Predict = Ctest;
                                    Labels_Predict = dtest;




                                    Best_C1 = 0;
                                    Best_C2 = 0;
                                    Best_C3 = 0;
                                    Best_C4 = 0;   


                                    Best_C1_ = 0;
                                    Best_C2_ = 0;
                                    Best_C3_ = 0;
                                    Best_C4_ = 0;

                                    Best_C1_1 = 0;
                                    Best_C2_1 = 0;
                                    Best_C3_1 = 0;
                                    Best_C4_1 = 0;



                                    C3_ACC = ones(1,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                                    C3_C1 = ones(length(C1_Interval),1);
                                    C3_C3 = ones(length(C3_Interval),1);
                                    Gam = ones(length(gamma_Interval),1);

                                    C3_ACC_ = ones(1,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                                    C3_C1_ = ones(length(C1_Interval),1);
                                    C3_C3_ = ones(length(C3_Interval),1);
                                    Gam_ = ones(length(gamma_Interval),1);

                                    C3_ACC_1 = ones(1,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                                    C3_C1_1 = ones(length(C1_Interval),1);
                                    C3_C3_1 = ones(length(C3_Interval),1);
                                    Gam_1 = ones(length(gamma_Interval),1);



                                    for ith_C3 = 1:length(C3_Interval) 
                                              C3 = C3_Interval(ith_C3);    % lambda3
                                              C4 = C3;

                                              Best_Acc = 0;%必须要清0，设为0后，其余参数就不需要再重置了。
                                              Best_Acc_ = 0;
                                              Best_Acc_1 = 0;

                                              for ith_C1 = 1:length(C1_Interval) 
                                                  C1 = C1_Interval(ith_C1);    % lambda3
                                                  C2 = C1;



                                                        for ith_gamma = 1:length(gamma_Interval)       %   gamma

                                                            Acc_SubPredict = zeros(1, 1);
                                                            Acc_SubPredict_ = zeros(1, 1);
                                                            Acc_SubPredict_1 = zeros(1, 1);


                                                            %%%%%%-------Computes the average distance between instances-------%%%%%%
                %                                                     M_Sub = size(Samples_Train, 1);
                %                                                     Index_Sub = combntns(1:M_Sub, 2); % Combination
                %                                                     delta_Sub = 0;
                %                                                     Num_Sub = size(Index_Sub, 1);
                %                                                     for i = 1:Num_Sub
                %                                                         delta_Sub = delta_Sub + norm(Samples_Train(Index_Sub(i, 1), :)-Samples_Train(Index_Sub(i, 2),:), 2)/Num_Sub;
                %                                                     end
                %                                                     %%%%%%-------Computes the average distance between instances-------%%%%%%
                                                            Kernel.gamma = gamma_Interval(ith_gamma);  %   gamma

                                                            I_A = Labels_Train == 1;
                                                            Samples_A = Samples_Train(I_A,:);
                                                            Labels_A = Labels_Train(I_A);

                                                            I_B = Labels_Train == -1;
                                                            Samples_B = Samples_Train(I_B,:);                        
                                                            Labels_B = Labels_Train(I_B);     


                                                            Parameter.ker = 'rbf';
                                                            Parameter.CC = C3;
                                                            Parameter.CR = C1;
                                                            Parameter.p1 = Kernel.gamma;
                                                            Parameter.algorithm = 'QP';
                                                            Parameter.showplots = false;  

                                                            Parameter_.ker = 'rbf';
                                                            Parameter_.CC = C3;
                                                            Parameter_.CR = C1;
                                                            Parameter_.p1 = Kernel.gamma;
                                                            Parameter_.algorithm = 'QP';
                                                            Parameter_.showplots = false; 

                                                            Parameter_1.ker = 'rbf';
                                                            Parameter_1.CC = C3;
                                                            Parameter_1.CR = C1;
                                                            Parameter_1.p1 = Kernel.gamma;
                                                            Parameter_1.algorithm = 'QP';
                                                            Parameter_1.showplots = false; 


                                                            [tbsvm_Substruct] = tbsvmtrain(Samples_Train,Labels_Train,Parameter);
                                                            [tbsvm_Substruct_] = IFtbsvmtrain_wang19(Samples_Train,Labels_Train,Parameter_);
                                                            [tbsvm_Substruct_1] = IFtbsvmtrain(Samples_Train,Labels_Train,Parameter_1);


                                                            [Acc]= tbsvmclass(tbsvm_Substruct,Samples_Predict,Labels_Predict);  
                                                            [Acc_]= tbsvmclass(tbsvm_Substruct_,Samples_Predict,Labels_Predict);   
                                                            [Acc_1]= tbsvmclass(tbsvm_Substruct_1,Samples_Predict,Labels_Predict);



                                                            Stop_Num = Stop_Num - 1;

                                                            disp([num2str(Stop_Num), ' step(s) remaining.'])


                                                            if Acc>Best_Acc
                                                                Best_Acc = Acc;
                                                                Best_C1 = C1;
                                                                Best_C3 = C3;                       
                                                                Best_Kernel = Kernel;
                                                            end

                                                            if Acc_>Best_Acc_
                                                                Best_Acc_ = Acc_;
                                                                Best_C1_ = C1;
                                                                Best_C3_ = C3;                       
                                                                Best_Kernel_ = Kernel;
                                                            end

                                                            if Acc_1>Best_Acc_1
                                                                Best_Acc_1 = Acc_1;
                                                                Best_C1_1 = C1;
                                                                Best_C3_1 = C3;                       
                                                                Best_Kernel_1 = Kernel;
                                                            end

                                                        end  % gamma

                                              end

                                            C3_ACC(ith_C3) = Best_Acc;
                                            C3_C1(ith_C3) = Best_C1;
                                            C3_C3(ith_C3) = Best_C3;
                                            Gam(ith_C3) = Best_Kernel.gamma;

                                            C3_ACC_(ith_C3) = Best_Acc_;
                                            C3_C1_(ith_C3) = Best_C1_;
                                            C3_C3_(ith_C3) = Best_C3_;
                                            Gam_(ith_C3) = Best_Kernel_.gamma;

                                            C3_ACC_1(ith_C3) = Best_Acc_1;
                                            C3_C1_1(ith_C3) = Best_C1_1;
                                            C3_C3_1(ith_C3) = Best_C3_1;
                                            Gam_1(ith_C3) = Best_Kernel_1.gamma;
                                    end

                                    [Best_Acc,i]=max(C3_ACC);
                                    [Best_Acc_,j]=max(C3_ACC_);
                                    [Best_Acc_1,k]=max(C3_ACC_1);




                                    Best_Parameter.ker = 'rbf';
                                    Best_Parameter.CC = C3_C3(i);
                                    Best_Parameter.CR = C3_C1(i);
                                    Best_Parameter.algorithm = 'QP';
                                    Best_Parameter.p1 = Gam(i);
                                    Best_Parameter.showplots = false; 


                                    Best_Parameter_.ker = 'rbf';
                                    Best_Parameter_.CC = C3_C3_(j);
                                    Best_Parameter_.CR = C3_C1_(j);
                                    Best_Parameter_.algorithm = 'QP';
                                    Best_Parameter_.p1 = Gam_(j);
                                    Best_Parameter_.showplots = false;

                                    Best_Parameter_1.ker = 'rbf';
                                    Best_Parameter_1.CC = C3_C3_1(k);
                                    Best_Parameter_1.CR = C3_C1_1(k);
                                    Best_Parameter_1.algorithm = 'QP';
                                    Best_Parameter_1.p1 = Gam_1(k);
                                    Best_Parameter_1.showplots = false;


                                    tic         
                                    [tbsvm_struct] = tbsvmtrain(Samples_Train,Labels_Train,Best_Parameter);
                                    t_Train(Times) = toc;

                                    tic         
                                    [tbsvm_struct_] = IFtbsvmtrain_wang19(Samples_Train,Labels_Train,Best_Parameter_);
                                    t_Train_(Times) = toc;

                                    tic         
                                    [tbsvm_struct_1] = IFtbsvmtrain(Samples_Train,Labels_Train,Best_Parameter_1);
                                    t_Train_1(Times) = toc;


                                   % Predict the data
                                    Acc_Predict(Times) = Best_Acc;
                                    Acc_Predict_(Times) = Best_Acc_;
                                    Acc_Predict_1(Times) = Best_Acc_1;

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
            pred_label = zeros(length(Samples_Predict),1);
            for i3 = 1:length(Samples_Predict)
                [max_val,max_ind] =  max(voting_mat(i3,:));
                pred_label(i3) = max_ind;
            end
            mul_Acc = (length(find(Labels_Predict == pred_label)))/length(Labels_Predict);             


                %%%%%%%---------------------Save the statiatics---------------------%%%%%%%
%                 Name = Str_Name(10:end-4);
                %                 Loc_Nam = [Location, '\', Name, '.txt'];
                %                 f = fopen(Loc_Nam, 'wt');
%                 disp('TBSVM');
%                 fprintf('%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train)), '.']);
%                 fprintf('%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(Acc_Predict)) '%.']);
% %                 fprintf('%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(100*Acc_Predict)), '.']);
% %                 fprintf('%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
% %                 fprintf('%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);
% 
%                 fprintf('%s\n', 'The Best_gamma is: ',num2str(Best_Kernel.gamma));
% %                 fprintf('%s\n', 'The Best_lambda1 is: ',num2str(Best_lambda1));
% %                 fprintf('%s\n','The Best_lambda2 is:',num2str(Best_lambda2));
% %                 fprintf('%s\n', 'The Best_lambda3 is: ',num2str(Best_lambda3));
% %                 fprintf('%s\n','The Best_lambda4 is:',num2str(Best_lambda4));
%                 fprintf('%s\n', 'The Best_C1 is: ',num2str(Best_C1));
%                 fprintf('%s\n','The Best_C2 is:',num2str(Best_C2));
%                 fprintf('%s\n', 'The Best_C3 is: ',num2str(Best_C3));
%                 fprintf('%s\n','The Best_C4 is:',num2str(Best_C4));

                temp = [mean(t_Train),mean(Acc_Predict),...
                   Best_Parameter.p1,Best_Parameter.CR,Best_Parameter.CC];
                xlswrite('result.xlsx',temp,['W',num2str(iData+2),':AA',num2str(iData+2)]);  
                xlswrite('result.xlsx',C3_ACC,['C',num2str(iData+123),':O',num2str(iData+123)]);

                temp_ = [mean(t_Train_),mean(Acc_Predict_),...
                   Best_Parameter_.p1,Best_Parameter_.CR,Best_Parameter_.CC];
                xlswrite('result.xlsx',temp_,['K',num2str(iData+2),':O',num2str(iData+2)]);  
                xlswrite('result.xlsx',C3_ACC_,['C',num2str(iData+63),':O',num2str(iData+63)]);

                temp_1 = [mean(t_Train_1),mean(Acc_Predict_1),...
                   Best_Parameter_1.p1,Best_Parameter_1.CR,Best_Parameter_1.CC];
                xlswrite('result.xlsx',temp_1,['Q',num2str(iData+2),':U',num2str(iData+2)]);  
                xlswrite('result.xlsx',C3_ACC_1,['C',num2str(iData+93),':O',num2str(iData+93)]);

            case 3   %%TSVM and FTSVM
            %% Train and predict the data

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

%                     Ctrain=awgn(Ctrain,0.1); % 高斯噪声
%                     Ctest=awgn(Ctest,0.1); % 高斯噪声
%                     Ctrain= svdatanorm(Ctrain,'svpline');
%                     Ctest= svdatanorm(Ctest,'scpline');
                    Samples_Train = Ctrain;
                    Labels_Train = dtrain;
                    Samples_Predict = Ctest;
                    Labels_Predict = dtest;

                    Best_C1 = 0;  

                    Best_C1_ = 0;              


                    C3_ACC = ones(1,length(C1_Interval));%由于xlwrite是写行向量，所以设置为行向量
                    C3_C1 = ones(length(C1_Interval),1);
                    Gam = ones(length(gamma_Interval),1);

                    C3_ACC_ = ones(1,length(C1_Interval));%由于xlwrite是写行向量，所以设置为行向量
                    C3_C1_ = ones(length(C1_Interval),1);
                    Gam_ = ones(length(gamma_Interval),1);
                      for ith_C1 = 1:length(C1_Interval) 
                           C1 = C1_Interval(ith_C1);    % lambda3
                           C2 = C1;

                           Best_Acc = 0;%必须要清0，设为0后，其余参数就不需要再重置了。
                           Best_Acc_ = 0;

                                    for ith_gamma = 1:length(gamma_Interval)       %   gamma

                                        Acc_SubPredict = zeros(1, 1);
                                        Acc_SubPredict_ = zeros(1, 1);


                                        %%%%%%-------Computes the average distance between instances-------%%%%%%
    %                                                     M_Sub = size(Samples_Train, 1);
    %                                                     Index_Sub = combntns(1:M_Sub, 2); % Combination
    %                                                     delta_Sub = 0;
    %                                                     Num_Sub = size(Index_Sub, 1);
    %                                                     for i = 1:Num_Sub
    %                                                         delta_Sub = delta_Sub + norm(Samples_Train(Index_Sub(i, 1), :)-Samples_Train(Index_Sub(i, 2),:), 2)/Num_Sub;
    %                                                     end
    %                                                     %%%%%%-------Computes the average distance between instances-------%%%%%%
                                        Kernel.gamma = gamma_Interval(ith_gamma);  %   gamma

                                        I_A = Labels_Train == 1;
                                        Samples_A = Samples_Train(I_A,:);
                                        Labels_A = Labels_Train(I_A);

                                        I_B = Labels_Train == -1;
                                        Samples_B = Samples_Train(I_B,:);                        
                                        Labels_B = Labels_Train(I_B);     

                                        C_s.C1 = C1;
                                        C_s.C2 = C2;
%                                         s = IFuzzy_MemberShip(Samples_Train, Labels_Train, Kernel, Best_u);
%     %                                                     C_s.C3 = C3;
%     %                                                     C_s.C4 = C4;
%                                         C_s.s1 = s.s1;
%                                         C_s.s2 = s.s2;

                                        C_s_.C1 = C1;
                                        C_s_.C2 = C2;
                                        s_ = Fuzzy_MemberShip(Samples_Train, Labels_Train, Kernel, Best_u);

                                        C_s_.s1 = s_.s1;
                                        C_s_.s2 = s_.s2;



                                        Outs_Train = Train_TSVM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train,C_s, Kernel, QPPs_Solver);
                                        Outs_Train_ = Train_FTSVM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train,C_s_, Kernel, QPPs_Solver);


                                        Acc = Predict_TSVM(Outs_Train, Samples_Predict,Labels_Predict, Samples_Train);   
                                        Acc_ = Predict_FTSVM(Outs_Train_, Samples_Predict,Labels_Predict, Samples_Train);


                                        Stop_Num = Stop_Num - 1;

                                        disp([num2str(Stop_Num), ' step(s) remaining.'])


                                        if Acc>Best_Acc
                                            Best_Acc = Acc;
                                            Best_C1 = C1;                       
                                            Best_Kernel = Kernel;
                                        end

                                        if Acc_>Best_Acc_
                                            Best_Acc_ = Acc_;
                                            Best_C1_ = C1;                     
                                            Best_Kernel_ = Kernel;
                                        end

                                    end  % gamma

                            C3_ACC(ith_C1) = Best_Acc;
                            C3_C1(ith_C1) = Best_C1;
                            Gam(ith_C1) = Best_Kernel.gamma;

                            C3_ACC_(ith_C1) = Best_Acc_;
                            C3_C1_(ith_C1) = Best_C1_;
                            Gam_(ith_C1) = Best_Kernel_.gamma;

                      end


                    [Best_Acc,i]=max(C3_ACC);
                    [Best_Acc_,j]=max(C3_ACC_);

                    BestC_s.C1 = C3_C1(i);
                    BestC_s.C2 = C3_C1(i);
                    Best_Kernel.gamma = Gam(i);
                    Best_Kernel.Type = 'RBF';
%                     Bests = IFuzzy_MemberShip(Samples_Train, Labels_Train, Best_Kernel, Best_u); 
%                     BestC_s.s1 = Bests.s1;
%                     BestC_s.s2 = Bests.s2;

                    BestC_s_.C1 = C3_C1_(j);
                    BestC_s_.C2 = C3_C1_(j);
                    Best_Kernel_.gamma = Gam_(j);
                    Best_Kernel_.Type = 'RBF';
                    Bests_ = Fuzzy_MemberShip(Samples_Train, Labels_Train, Best_Kernel_, Best_u); 
                    BestC_s_.s1 = Bests_.s1;
                    BestC_s_.s2 = Bests_.s2;




                    tic         
                    Outs_Train = Train_TSVM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, BestC_s, Best_Kernel, QPPs_Solver);
                    t_Train(Times) = toc;

                    tic         
                    Outs_Train_ = Train_FTSVM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, BestC_s_, Best_Kernel_, QPPs_Solver);
                    t_Train_(Times) = toc;

                   % Predict the data
                    Acc_Predict(Times) = Best_Acc;
                    Acc_Predict_(Times) = Best_Acc_;

                end



                temp = [mean(t_Train),mean(100*Acc_Predict),...
                   Best_Kernel.gamma,BestC_s.C1];
                xlswrite('result.xlsx',temp,['AC',num2str(iData+2),':AF',num2str(iData+2)]);
                xlswrite('result.xlsx',100*C3_ACC,['C',num2str(iData+153),':O',num2str(iData+153)]);

                temp_ = [mean(t_Train_),mean(100*Acc_Predict_),...
                   Best_Kernel_.gamma,BestC_s_.C1];
                xlswrite('result.xlsx',temp_,['AH',num2str(iData+2),':AK',num2str(iData+2)]);
                xlswrite('result.xlsx',100*C3_ACC_,['C',num2str(iData+183),':O',num2str(iData+183)]);

           case 4   %%SVM and FSVM
            %% Train and predict the data

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

%                     Ctrain=awgn(Ctrain,0.1); % 高斯噪声
%                     Ctest=awgn(Ctest,0.1); % 高斯噪声
%                     Ctrain= svdatanorm(Ctrain,'svpline');
%                     Ctest= svdatanorm(Ctest,'scpline');
                    Samples_Train = Ctrain;
                    Labels_Train = dtrain;
                    Samples_Predict = Ctest;
                    Labels_Predict = dtest; 


                    Best_C1 = 0;
                    Best_C1_ = 0;



                    C3_ACC = ones(1,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                    C3_C1 = ones(length(C1_Interval),1);
                    Gam = ones(length(gamma_Interval),1);

                    C3_ACC_ = ones(1,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                    C3_C1_ = ones(length(C1_Interval),1);
                    Gam_ = ones(length(gamma_Interval),1);



                      for ith_C1 = 1:length(C1_Interval) 
                           C1 = C1_Interval(ith_C1);    % lambda3

                           Best_Acc = 0;%必须要清0，设为0后，其余参数就不需要再重置了。
                           Best_Acc_ = 0;

                                    for ith_gamma = 1:length(gamma_Interval)       %   gamma

                                        Acc_SubPredict = zeros(1, 1);
                                        Acc_SubPredict_ = zeros(1, 1);


                                        %%%%%%-------Computes the average distance between instances-------%%%%%%
    %                                                     M_Sub = size(Samples_Train, 1);
    %                                                     Index_Sub = combntns(1:M_Sub, 2); % Combination
    %                                                     delta_Sub = 0;
    %                                                     Num_Sub = size(Index_Sub, 1);
    %                                                     for i = 1:Num_Sub
    %                                                         delta_Sub = delta_Sub + norm(Samples_Train(Index_Sub(i, 1), :)-Samples_Train(Index_Sub(i, 2),:), 2)/Num_Sub;
    %                                                     end
    %                                                     %%%%%%-------Computes the average distance between instances-------%%%%%%
                                        Kernel.gamma = gamma_Interval(ith_gamma);  %   gamma

                                        I_A = Labels_Train == 1;
                                        Samples_A = Samples_Train(I_A,:);
                                        Labels_A = Labels_Train(I_A);

                                        I_B = Labels_Train == -1;
                                        Samples_B = Samples_Train(I_B,:);                        
                                        Labels_B = Labels_Train(I_B); 

%                                         s = Fuzzy_MemberShip(Samples_Train,Labels_Train, Kernel, Best_u);
                                        s_ = Fuzzy_MemberShip_(Samples_Train,Labels_Train, Kernel, Best_u);


                                        Outs_SubTrain =  Train_SVM(Samples_Train, Labels_Train, C1*abs(Labels_Train), Kernel, QPPs_Solver);
                                        Outs_SubTrain_ =  Train_SVM(Samples_Train, Labels_Train, C1*s_, Kernel, QPPs_Solver);

                                        Acc = Predict_SVM(Outs_SubTrain, Samples_Predict, Labels_Predict);  
                                        Acc_ = Predict_SVM(Outs_SubTrain_, Samples_Predict, Labels_Predict); 


                                        Stop_Num = Stop_Num - 1;

                                        disp([num2str(Stop_Num), ' step(s) remaining.'])


                                        if Acc>Best_Acc
                                            Best_Acc = Acc;
                                            Best_C1 = C1;                     
                                            Best_Kernel = Kernel;
                                        end

                                        if Acc_>Best_Acc_
                                            Best_Acc_ = Acc_;
                                            Best_C1_ = C1;                     
                                            Best_Kernel_ = Kernel;
                                        end


                                    end  % gamma
                                    C3_ACC(ith_C1) = Best_Acc;
                                    C3_C1(ith_C1) = Best_C1;
                                    Gam(ith_C1) = Best_Kernel.gamma;

                                    C3_ACC_(ith_C1) = Best_Acc_;
                                    C3_C1_(ith_C1) = Best_C1_;
                                    Gam_(ith_C1) = Best_Kernel_.gamma;
                      end

                    [Best_Acc,i]=max(C3_ACC);
                    [Best_Acc_,j]=max(C3_ACC_);

                    Best_C1 = C3_C1(i);
                    Best_Kernel.gamma = Gam(i);
%                     Bests = Fuzzy_MemberShip_(Samples_Train, Labels_Train, Best_Kernel, Best_u); 



                    Best_C1_ = C3_C1_(i);
                    Best_Kernel_.gamma = Gam_(j);
                    Bests_ = Fuzzy_MemberShip_(Samples_Train, Labels_Train, Best_Kernel, Best_u); 


                    tic         
                    Outs_Train =  Train_SVM(Samples_Train, Labels_Train, Best_C1*abs(Labels_Train), Best_Kernel, QPPs_Solver);
                    t_Train(Times) = toc;

                    tic         
                    Outs_Train_ =  Train_SVM(Samples_Train, Labels_Train, Best_C1_*Bests_, Best_Kernel_, QPPs_Solver);
                    t_Train_(Times) = toc;

                   % Predict the data
                    Acc_Predict(Times) = Best_Acc;
                    Acc_Predict_(Times) = Best_Acc_;

                end


%             
                temp = [mean(t_Train),mean(100*Acc_Predict),...
                   Best_Kernel.gamma,Best_C1];
                xlswrite('result.xlsx',temp,['AM',num2str(iData+2),':AP',num2str(iData+2)]);
                xlswrite('result.xlsx',100*C3_ACC,['C',num2str(iData+213),':O',num2str(iData+213)]);

                temp = [mean(t_Train_),mean(100*Acc_Predict_),...
                   Best_Kernel_.gamma,Best_C1_];
                xlswrite('result.xlsx',temp,['AR',num2str(iData+2),':AU',num2str(iData+2)]);
                xlswrite('result.xlsx',100*C3_ACC_,['C',num2str(iData+243),':O',num2str(iData+243)]);

            case 5  %%LDM and FLDM
            %% Train and predict the data

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
% 
%                     Ctrain1=awgn(Ctrain,0.1); % 高斯噪声
%                     Ctest1=awgn(Ctest,0.1); % 高斯噪声
%                     Ctrain= svdatanorm(Ctrain,'svpline');
%                     Ctest= svdatanorm(Ctest,'scpline');
                    Samples_Train = Ctrain;
                    Labels_Train = dtrain;
                    Samples_Predict = Ctest;
                    Labels_Predict = dtest;

                    Best_lambda1 = 0;
                    Best_lambda2 = 0;
                    Best_C1 = 0;

                    Best_lambda1_ = 0;
                    Best_lambda2_ = 0;
                    Best_C1_ = 0;

                    C3_ACC = ones(1,length(C1_Interval));%由于xlwrite是写行向量，所以设置为行向量
                    C3_lambda1 = ones(length(lambda1_Interval),1);
                    C3_lambda2 = ones(length(lambda2_Interval),1);
                    C3_C1 = ones(length(C1_Interval),1);
                    Gam = ones(length(gamma_Interval),1);

                    C3_ACC_ = ones(1,length(C1_Interval));%由于xlwrite是写行向量，所以设置为行向量
                    C3_lambda1_ = ones(length(lambda1_Interval),1);
                    C3_lambda2_ = ones(length(lambda2_Interval),1);
                    C3_C1_ = ones(length(C1_Interval),1);
                    Gam_ = ones(length(gamma_Interval),1);

                      for ith_C1 = 1:length(C1_Interval)                           
                          C1 = C1_Interval(ith_C1);    % lambda3

                          Best_Acc = 0;%必须要清0，设为0后，其余参数就不需要再重置了。
                           Best_Acc_ = 0;

                            for ith_lambda1 = 1:length(lambda1_Interval)    % lambda1
                                lambda1 = lambda1_Interval(ith_lambda1);    % lambda1
                                for ith_lambda2 = 1:length(lambda2_Interval)
                                  lambda2 = lambda2_Interval(ith_lambda2);



                                                        for ith_gamma = 1:length(gamma_Interval)       %   gamma

                                                            Acc_SubPredict = zeros(1, 1);
                                                            Acc_SubPredict_ = zeros(1, 1);


                                                            %%%%%%-------Computes the average distance between instances-------%%%%%%
        %                                                     M_Sub = size(Samples_Train, 1);
        %                                                     Index_Sub = combntns(1:M_Sub, 2); % Combination
        %                                                     delta_Sub = 0;
        %                                                     Num_Sub = size(Index_Sub, 1);
        %                                                     for i = 1:Num_Sub
        %                                                         delta_Sub = delta_Sub + norm(Samples_Train(Index_Sub(i, 1), :)-Samples_Train(Index_Sub(i, 2),:), 2)/Num_Sub;
        %                                                     end
        %                                                     %%%%%%-------Computes the average distance between instances-------%%%%%%
                                                            Kernel.gamma = gamma_Interval(ith_gamma);  %   gamma

                                                            I_A = Labels_Train == 1;
                                                            Samples_A = Samples_Train(I_A,:);
                                                            Labels_A = Labels_Train(I_A);

                                                            I_B = Labels_Train == -1;
                                                            Samples_B = Samples_Train(I_B,:);                        
                                                            Labels_B = Labels_Train(I_B);     


                                                            C_s_.C = C1*abs(Labels_Train);
                                                            C_s_.s = Fuzzy_MemberShip_(Samples_Train, Labels_Train,Kernel,Best_u);


                                                            Outs_SubTrain = Train_LDM(Samples_Train, Labels_Train, lambda1, lambda2, C1*abs(Labels_Train),Kernel, QPPs_Solver);
                                                            Outs_SubTrain_ = Train_FLDM(Samples_Train, Labels_Train, lambda1, lambda2, C_s_, FLDM_Type, Kernel, QPPs_Solver);

                                                            Acc = Predict_LDM(Outs_SubTrain, Samples_Predict, Labels_Predict);  
                                                            Acc_ = Predict_FLDM(Outs_SubTrain_, Samples_Predict, Labels_Predict);


                                                            Stop_Num = Stop_Num - 1;

                                                            disp([num2str(Stop_Num), ' step(s) remaining.'])


                                                            if Acc>Best_Acc
                                                                Best_Acc = Acc;
                                                                Best_lambda1 = lambda1;
                                                                Best_lambda2 = lambda2;
                                                                Best_C1 = C1;                     
                                                                Best_Kernel = Kernel;
                                                            end

                                                             if Acc_>Best_Acc_
                                                                Best_Acc_ = Acc_;
                                                                Best_lambda1_ = lambda1;
                                                                Best_lambda2_ = lambda2;
                                                                Best_C1_ = C1;                     
                                                                Best_Kernel_ = Kernel;
                                                            end

                                                        end  % gamma

                                end
                            end    % lambda1
                            C3_ACC(ith_C1) = Best_Acc;
                            C3_lambda1(ith_C1) = Best_lambda1;
                            C3_lambda2(ith_C1) = Best_lambda2;
                            C3_C1(ith_C1) = Best_C1;
                            Gam(ith_C1) = Best_Kernel.gamma;

                            C3_ACC_(ith_C1) = Best_Acc_;
                            C3_lambda1_(ith_C1) = Best_lambda1_;
                            C3_lambda2_(ith_C1) = Best_lambda2_;
                            C3_C1_(ith_C1) = Best_C1_;
                            Gam_(ith_C1) = Best_Kernel_.gamma;
                      end
                      [Best_Acc,i]=max(C3_ACC);
                      [Best_Acc_,j]=max(C3_ACC_);

                        Best_C1 = C3_C1(i);
                        Best_lambda1 = C3_lambda1(i);
                        Best_lambda2 = C3_lambda2(i);
                        Best_Kernel.gamma = Gam(i);
%                         BestC_s.C = Best_C1*abs(Labels_Train);
%                         BestC_s.s = IFuzzy_MemberShip_wang19(Samples_Train, Labels_Train,Best_Kernel,Best_u);

                        Best_C1_ = C3_C1_(j);
                        Best_lambda1_ = C3_lambda1_(j);
                        Best_lambda2_ = C3_lambda2_(j);
                        Best_Kernel_.gamma = Gam_(j);
                        BestC_s_.C = Best_C1_*abs(Labels_Train);
                        BestC_s_.s = Fuzzy_MemberShip_(Samples_Train, Labels_Train,Best_Kernel_,Best_u);




                    tic         
                    Outs_SubTrain = Train_LDM(Samples_Train, Labels_Train, Best_lambda1, Best_lambda2, Best_C1*abs(Labels_Train), Best_Kernel, QPPs_Solver);
                    t_Train(Times) = toc;

                    tic         
                    Outs_SubTrain_ = Train_FLDM(Samples_Train, Labels_Train, Best_lambda1_, Best_lambda2_, BestC_s_, FLDM_Type, Best_Kernel_, QPPs_Solver);
                    t_Train_(Times) = toc;


                   % Predict the data
                    Acc_Predict(Times) = Best_Acc;
                    Acc_Predict_(Times) = Best_Acc_;

                end


                %%%%%%%---------------------Save the statiatics---------------------%%%%%%%
%                 Name = Str_Name(10:end-4);
                %                 Loc_Nam = [Location, '\', Name, '.txt'];
                %                 f = fopen(Loc_Nam, 'wt');
                disp('LDM');

                temp = [mean(t_Train),mean(100*Acc_Predict),...
                   Best_Kernel.gamma,Best_lambda1,Best_lambda2,Best_C1];
                xlswrite('result.xlsx',temp,['AW',num2str(iData+2),':BB',num2str(iData+2)]); 
                xlswrite('result.xlsx',100*C3_ACC,['C',num2str(iData+273),':O',num2str(iData+273)]); 

                temp_ = [mean(t_Train_),mean(100*Acc_Predict_),...
                   Best_Kernel_.gamma,Best_lambda1_,Best_lambda2_,Best_C1_];
                xlswrite('result.xlsx',temp_,['BD',num2str(iData+2),':BI',num2str(iData+2)]); 
                xlswrite('result.xlsx',100*C3_ACC_,['C',num2str(iData+303),':O',num2str(iData+303)]);       

        end
    end
 
