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
gamma_Interval = 2.^(-5:2);
lambda1_Interval = 2.^(-8:4);
lambda2_Interval = 2.^(-8:4);
C1_Interval = 2.^(-8:4);
C3_Interval = 2.^(-8:4);

% gamma_Interval = 2.^(-5:-5);
% lambda1_Interval = 2.^(-8:-8);
% lambda2_Interval = 2.^(-8:-8);
% C1_Interval = 2.^(-8:-8);
% C3_Interval = 2.^(-8:-8);


Best_u = 0.4;


%% Counts
N_Times = 5;
K_fold = 5;
TrainRate = 0.9;
switch Kernel.Type
    case 'Linear'
        Stop_Num = 7*N_Times*length(lambda1_Interval)*length(lambda2_Interval)*length(C1_Interval)*length(C3_Interval) + 1;
    case 'RBF'
        Stop_Num = 1*N_Times*K_fold*length(lambda1_Interval)*length(lambda2_Interval)*length(C1_Interval)*length(C3_Interval)*length(gamma_Interval) + 1;
        
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
for iData =1:1
%                 Str_Name = Str(iData).Name;
%                 Output = load(Str_Name);
%                 Data_Name = fieldnames(Output);   % A struct data
%                 Data_Original = getfield(Output, Data_Name{1}); % Abstract the data
%                 % Normalization
%                 Data_Original = [mapminmax(Data_Original(:, 1:end-1)', 0, 1)', Data_Original(:, end)]; % Map the original data to value between [0, 1] by colum
% %                 M_Original = size(Data_Original, 1);
% %                 Data_Original = Data_Original(randperm(M_Original), :);

             if (iData==1)
                load('Data_mat_x\waveform-5000_1_2.mat');
                X = file_data;
                % Normalization
                X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
                M_Original = size(X, 1);
                X = X(randperm(M_Original), :);     

             end      
             
             if (iData==2)
                load('Data_mat_x\waveform-5000_0_2.mat');
                X = file_data;
                % Normalization
                X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
                M_Original = size(X, 1);
                X = X(randperm(M_Original), :);     

             end
             
             if (iData==3)
                load('Data_mat_x\waveform-5000_0_1.mat');
                X = file_data;
                % Normalization
                X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
                M_Original = size(X, 1);
                X = X(randperm(M_Original), :);     

             end
             
             if (iData==4)
                load('Data_mat_x\spambase.mat');
                X = file_data;
                % Normalization
                X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
                M_Original = size(X, 1);
                X = X(randperm(M_Original), :);     

             end
             
             if (iData==5)
                load('Data_mat_x\mushroom.mat');
                X = file_data;
                % Normalization
                X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
                M_Original = size(X, 1);
                X = X(randperm(M_Original), :);     

             end
             
             if (iData==6)
                load('Data_mat_x\kr-vs-kp.mat');
                X = file_data;
                % Normalization
                X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
                M_Original = size(X, 1);
                X = X(randperm(M_Original), :);     

             end
             
             if (iData==7)
                load('Data_mat_x\credit-g.mat');
                X = file_data;
                % Normalization
                X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
                M_Original = size(X, 1);
                X = X(randperm(M_Original), :);     

             end
             
             if (iData==8)
                load('Data_mat_x\credit6000_126.mat');
                X = file_data;
                % Normalization
                X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
                M_Original = size(X, 1);
                X = X(randperm(M_Original), :);     

             end
             
             if(iData==9)
                load('Data_mat\CMC.mat');
                X= CMC;
                X(find(X(:,end)~=1),end)=-1;
                % Normalization
                X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
                M_Original = size(X, 1);
                X = X(randperm(M_Original), :);             

             end




    
    

    
      
    for type=2:5
        switch type
           case 1   %%DC-IFTBLDM and TBLDM
            %% Train and predict the data

                %%%%%%%-----------------Training the best parameters-----------------%%%%%%%
                t_Train = zeros(N_Times,length(C3_Interval));
                t_Train_Predict = zeros(N_Times, 1);
%                 t_Train_ = zeros(N_Times, 1);
                Acc_Predict = zeros(N_Times, 1);
%                 Acc_Predict_ = zeros(N_Times, 1);
                
%                 Accuracy = zeros();
                
                C3_ACC = zeros(N_Times,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_ACCp = zeros(N_Times,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_lambda1 = zeros(N_Times,length(lambda1_Interval));
                C3_lambda2 = zeros(N_Times, length(lambda2_Interval));
                C3_C1 = zeros(N_Times, length(C1_Interval));
                C3_C3 = zeros(N_Times, length(C3_Interval));
                Gam = zeros(N_Times,length(gamma_Interval));
                Acc_Leader = 0;
                
%                 n = 0;
                
                for Times = 1: N_Times
                    
%                    n = n+1;
% 
%                    if n == 1
%                         n = 1;
%                     end
%                     if n == 2
%                         n = 2;
%                     end
%                     if n == 3
%                         n = 3;
%                     end
%                     if n == 4
%                         n = 4;
%                     end
%                     if n == 5
%                         n = 5;
%                     end
%                     
%                     
%                                           
                    [test, train] = Data_Rate(X, TrainRate); 
                    
                    

                    Ctrain=train(:,1:end-1);
                    dtrain= train(:,end);
                    Ctest= test(:,1:end-1);
                    dtest= test(:,end);

                    Samples_Train = Ctrain;
                    Samples_Predict = Ctest;
                    Labels_Train = dtrain;                    
                    Labels_Predict = dtest;                    
                                                           
                    
 

                    
%                     Best_lambda1_ = 0;
%                     Best_lambda2_ = 0;
%                     Best_lambda3_ = 0;
%                     Best_lambda4_ = 0;
%                     Best_C1_ = 0;
%                     Best_C2_ = 0;
%                     Best_C3_ = 0;
%                     Best_C4_ = 0;  
                    
                    

                    
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
%                             Best_Acc_ = 0;
                                                          
                            
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
                                                    
                                                    Indices = crossvalind('Kfold', length(Labels_Train),K_fold);
                                                    
                                                    
                                                    Acc_SubPredict = zeros(K_fold, 1);
                                                    for repeat = 1:K_fold
                                                            I_SubTrain = ~(Indices == repeat);
                                                            Samples_SubTrain = Samples_Train(I_SubTrain,:);
                                                            Labels_SubTrain = Labels_Train(I_SubTrain,:);
                                                            
                                                            I_SubPredict = ~I_SubTrain;
                                                            Samples_SubPredict = Samples_Train(I_SubPredict,:);
                                                            Labels_SubPredict = Labels_Train(I_SubPredict,:);

                                                            
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

                                                            I_SubA = Labels_SubTrain == 1;
                                                            Samples_SubA = Samples_SubTrain(I_SubA,:);
                                                            Labels_SubA = Labels_SubTrain(I_SubA);

                                                            I_SubB = Labels_SubTrain == -1;
                                                            Samples_SubB = Samples_SubTrain(I_SubB,:);                        
                                                            Labels_SubB = Labels_SubTrain(I_SubB);    



%                                                             membership(Samples_Train', Labels_Train')
                                                            s = IFuzzy_MemberShip(Samples_SubTrain, Labels_SubTrain, Kernel, Best_u); 
                                                            C_s.C1 = C1;
                                                            C_s.C2 = C2;
                                                            C_s.s1 = s.s1;
                                                            C_s.s2 = s.s2;
                                                            C_s.C3 = C3;
                                                            C_s.C4 = C4;



                                                            Outs_SubTrain = Train_FTBLDM(Samples_SubA, Labels_SubA, Samples_SubB,Labels_SubB, Samples_SubTrain, lambda1,lambda2 , C_s, Kernel, QPPs_Solver);
%                                                             Outs_Train_ = Train_TBLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, lambda1,lambda2 , C_s, Kernel, QPPs_Solver);
                                                            SubAcc = Predict_FTBLDM(Outs_SubTrain, Samples_SubPredict,Labels_SubPredict, Samples_SubTrain);   
% %                                                             Acc_ = Predict_TBLDM(Outs_Train_, Samples_Predict,Labels_Predict, Samples_Train);

                                                           Acc_SubPredict(repeat) = SubAcc;
                                                           Stop_Num = Stop_Num - 1;

                                                            disp([num2str(Stop_Num), ' step(s) remaining.'])
                                                    end


                                                    if mean(Acc_SubPredict) > Best_Acc
                                                        Best_Acc = mean(Acc_SubPredict);
                                                        Best_lambda1 = lambda1;
                                                        Best_lambda2 = lambda2; 
                                                        Best_C1 = C1;
                                                        Best_C3 = C3;                      
                                                        Best_Kernel = Kernel;
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
                            
                                                       
                            C3_ACC(Times, ith_C3) = Best_Acc;
                            C3_lambda1(Times, ith_C3) = Best_lambda1;
                            C3_lambda2(Times, ith_C3) = Best_lambda2;
                            C3_C1(Times, ith_C3) = Best_C1;
                            C3_C3(Times, ith_C3) = Best_C3;
                            Gam(Times, ith_C3) = Best_Kernel.gamma;
                            
                            I_A = Labels_Train == 1;
                            Samples_A = Samples_Train(I_A,:);
                            Labels_A = Labels_Train(I_A,:);

                            I_B = Labels_Train == -1;
                            Samples_B = Samples_Train(I_B,:);
                            Labels_B = Labels_Train(I_B,:);  
                            
                            BestC_s.C1 = Best_C1;
                            BestC_s.C2 = Best_C1;
                            Bests = IFuzzy_MemberShip(Samples_Train, Labels_Train, Best_Kernel, Best_u);  %!!!!
                            BestC_s.s1 = Bests.s1;
                            BestC_s.s2 = Bests.s2;
                            BestC_s.C3 = Best_C3;
                            BestC_s.C4 = Best_C3;
                            
                            tic
                            Outs_Train = Train_FTBLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, Best_lambda1, Best_lambda2, BestC_s, Best_Kernel, QPPs_Solver);
                            t_Train(Times, ith_C3) = toc;
                            
                            Acc_Predictp = Predict_FTBLDM(Outs_Train, Samples_Predict,Labels_Predict, Samples_Train);
                            C3_ACCp(Times, ith_C3) = Acc_Predictp;                           
                            
                    end

                    [Best_Acc,i]=max(C3_ACC(Times,:));
%                     [Best_Acc_,j]=max(C3_ACC_);
                     
                                      
                    
                    
                    Acc_Predict(Times) = C3_ACCp(Times,i);
                    t_Train_Predict(Times) = t_Train(Times,i);
                    
                    
                    
                    if Acc_Predict(Times)>Acc_Leader
                        Acc_Leader = Acc_Predict(Times);
                        BestC_s.C1 = C3_C1(Times,i);
                        BestC_s.C2 = C3_C1(Times,i);
                        BestC_s.C3 = C3_C3(Times,i);
                        BestC_s.C4 = C3_C3(Times,i);
                        Best_lambda1 = C3_lambda1(Times,i);
                        Best_lambda2 = C3_lambda2(Times,i);
                        Best_Kernel.gamma = Gam(Times,i);
                        Bests = IFuzzy_MemberShip(Samples_Train, Labels_Train, Best_Kernel, Best_u); 
                        BestC_s.s1 = Bests.s1;
                        BestC_s.s2 = Bests.s2;
                    end




                end
             

                temp = [mean(t_Train_Predict),100*mean(Acc_Predict),100*std(Acc_Predict),100*Acc_Leader,...
                   Best_Kernel.gamma, Best_lambda1,Best_lambda2,BestC_s.C1,BestC_s.C3];
                xlswrite('result.xlsx',temp,['C',num2str(iData+2),':K',num2str(iData+2)]);
                
                xlswrite('result.xlsx',100*(C3_ACCp(1,:)),['C',num2str(iData+44),':O',num2str(iData+44)]);
                xlswrite('result.xlsx',100*(C3_ACCp(2,:)),['Q',num2str(iData+44),':AC',num2str(iData+44)]);
                xlswrite('result.xlsx',100*(C3_ACCp(3,:)),['AE',num2str(iData+44),':AQ',num2str(iData+44)]);
                xlswrite('result.xlsx',100*(C3_ACCp(4,:)),['AS',num2str(iData+44),':BE',num2str(iData+44)]);
                xlswrite('result.xlsx',100*(C3_ACCp(5,:)),['BG',num2str(iData+44),':BS',num2str(iData+44)]);
                xlswrite('result.xlsx',100*mean(C3_ACCp),['BU',num2str(iData+44),':CG',num2str(iData+44)]);
                
                
                xlswrite('result.xlsx',100*Acc_Predict',['CI',num2str(iData+44),':CM',num2str(iData+44)]);%xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的

%                 temp_ = [mean(t_Train_),mean(100*Acc_Predict_),...
%                    Best_Kernel_.gamma, Best_lambda1_,Best_lambda2_,BestC_s_.C1,BestC_s_.C3];
%                 xlswrite('result.xlsx',temp_,['K',num2str(iData+2),':Q',num2str(iData+2)]); 
%                 xlswrite('result.xlsx',C3_ACC_,['C',num2str(iData+52),':O',num2str(iData+52)]);

                
            case 2  %%TBSVM and IFTSVM and CDFSVM
            %% Train and predict the data

                %%%%%%%-----------------Training the best parameters-----------------%%%%%%%
                t_Train = zeros(N_Times,length(C3_Interval));
                t_Train_ = zeros(N_Times,length(C3_Interval));
                t_Train_1 = zeros(N_Times,length(C3_Interval));
                
                Acc_Predict = zeros(N_Times, 1);
                Acc_Predict_ = zeros(N_Times, 1);
                Acc_Predict_1 = zeros(N_Times, 1);
                
                t_Train_Predict = zeros(N_Times, 1);
                t_Train_Predict_ = zeros(N_Times, 1);
                t_Train_Predict_1 = zeros(N_Times, 1);
                
%                 MarginMEAN_Train = zeros(N_Times, 1);
%                 MarginSTD_Train = zeros(N_Times, 1);
%                 Accuracy = zeros();

                C3_ACC = zeros(N_Times,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_ACCp = zeros(N_Times,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_C1 = zeros(N_Times,length(C1_Interval));
                C3_C3 = zeros(N_Times,length(C3_Interval));
                Gam = zeros(N_Times,length(gamma_Interval));

                C3_ACC_ = zeros(N_Times,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_ACCp_ = zeros(N_Times,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_C1_ = zeros(N_Times,length(C1_Interval));
                C3_C3_ = zeros(N_Times,length(C3_Interval));
                Gam_ = zeros(N_Times,length(gamma_Interval));

                C3_ACC_1 = zeros(N_Times,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_ACCp_1 = zeros(N_Times,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_C1_1 = zeros(N_Times,length(C1_Interval));
                C3_C3_1 = zeros(N_Times,length(C3_Interval));
                Gam_1 = zeros(N_Times,length(gamma_Interval));
                
                Acc_Leader = 0;
                Acc_Leader_ = 0;
                Acc_Leader_1 = 0;
                
                Times_Leader = 0;
                Times_Leader_ = 0;
                Times_Leader_1 = 0;
                i_Leader = 0;
                j_Leader = 0;
                k_Leader = 0;
                    
                for Times = 1: N_Times
                    
                    [test, train] = Data_Rate(X, TrainRate);        
                    Ctrain=train(:,1:end-1);
                    dtrain= train(:,end);
                    Ctest= test(:,1:end-1);
                    dtest= test(:,end);

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
                                      
                                        Indices = crossvalind('Kfold', length(Labels_Train),K_fold);
                                        Acc_SubPredict = zeros(K_fold, 1);
                                        Acc_SubPredict_ = zeros(K_fold, 1);
                                        Acc_SubPredict_1 = zeros(K_fold, 1);                                      
                                       for repeat = 1:K_fold
                                           
                                            I_SubTrain = ~(Indices == repeat);
                                            Samples_SubTrain = Samples_Train(I_SubTrain,:);
                                            Labels_SubTrain = Labels_Train(I_SubTrain,:);

                                            I_SubPredict = ~I_SubTrain;
                                            Samples_SubPredict = Samples_Train(I_SubPredict,:);
                                            Labels_SubPredict = Labels_Train(I_SubPredict,:);                                        




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


                                            [Subtbsvm_Substruct] = tbsvmtrain(Samples_SubTrain,Labels_SubTrain,Parameter);
                                            [Subtbsvm_Substruct_] = IFtbsvmtrain_wang19(Samples_SubTrain,Labels_SubTrain,Parameter_);
                                            [Subtbsvm_Substruct_1] = ftbsvmtrain(Samples_SubTrain,Labels_SubTrain,Parameter_1);
                                            

                                            [SubAcc]= tbsvmclass(Subtbsvm_Substruct,Samples_SubPredict,Labels_SubPredict);  
                                            [SubAcc_]= tbsvmclass(Subtbsvm_Substruct_,Samples_SubPredict,Labels_SubPredict);   
                                            [SubAcc_1]= tbsvmclass(Subtbsvm_Substruct_1,Samples_SubPredict,Labels_SubPredict);



                                            Acc_SubPredict(repeat) = SubAcc;
                                            Acc_SubPredict_(repeat) = SubAcc_;
                                            Acc_SubPredict_1(repeat) = SubAcc_1;
                                            Stop_Num = Stop_Num - 1;

                                            disp([num2str(Stop_Num), ' step(s) remaining.'])


                                            if mean(Acc_SubPredict)>Best_Acc
                                                Best_Acc = mean(Acc_SubPredict);
                                                Best_C1 = C1;
                                                Best_C3 = C3;                       
                                                Best_Kernel = Kernel;
                                            end
                                       
                                            
                                            if mean(Acc_SubPredict_)>Best_Acc_
                                                Best_Acc_ = mean(Acc_SubPredict_);
                                                Best_C1_ = C1;
                                                Best_C3_ = C3;                       
                                                Best_Kernel_ = Kernel;
                                            end
                                            
                                            if mean(Acc_SubPredict_1)>Best_Acc_1
                                                Best_Acc_1 = mean(Acc_SubPredict_1);
                                                Best_C1_1 = C1;
                                                Best_C3_1 = C3;                       
                                                Best_Kernel_1 = Kernel;
                                            end
                                       end
                                  end
                                 
                              end
                              
                            C3_ACC(Times,ith_C3) = Best_Acc;
                            C3_C1(Times,ith_C3) = Best_C1;
                            C3_C3(Times,ith_C3) = Best_C3;
                            Gam(Times,ith_C3) = Best_Kernel.gamma;

                            C3_ACC_(Times,ith_C3) = Best_Acc_;
                            C3_C1_(Times,ith_C3) = Best_C1_;
                            C3_C3_(Times,ith_C3) = Best_C3_;
                            Gam_(Times,ith_C3) = Best_Kernel_.gamma;
                            
                            C3_ACC_1(Times,ith_C3) = Best_Acc_1;
                            C3_C1_1(Times,ith_C3) = Best_C1_1;
                            C3_C3_1(Times,ith_C3) = Best_C3_1;
                            Gam_1(Times,ith_C3) = Best_Kernel_1.gamma;
                            
                            Best_Parameter.ker = 'rbf';
                            Best_Parameter.CC = C3_C3(Times,ith_C3);
                            Best_Parameter.CR = C3_C1(Times,ith_C3);
                            Best_Parameter.algorithm = 'QP';
                            Best_Parameter.p1 = Gam(Times,ith_C3);
                            Best_Parameter.showplots = false; 


                            Best_Parameter_.ker = 'rbf';
                            Best_Parameter_.CC = C3_C3_(Times,ith_C3);
                            Best_Parameter_.CR = C3_C1_(Times,ith_C3);
                            Best_Parameter_.algorithm = 'QP';
                            Best_Parameter_.p1 = Gam_(Times,ith_C3);
                            Best_Parameter_.showplots = false;

                            Best_Parameter_1.ker = 'rbf';
                            Best_Parameter_1.CC = C3_C3_1(Times,ith_C3);
                            Best_Parameter_1.CR = C3_C1_1(Times,ith_C3);
                            Best_Parameter_1.algorithm = 'QP';
                            Best_Parameter_1.p1 = Gam_1(Times,ith_C3);
                            Best_Parameter_1.showplots = false;                            

                            
                            tic         
                            [tbsvm_struct] = tbsvmtrain(Samples_Train,Labels_Train,Best_Parameter);
                            t_Train(Times,ith_C3) = toc;

                            tic         
                            [tbsvm_struct_] = IFtbsvmtrain_wang19(Samples_Train,Labels_Train,Best_Parameter_);
                            t_Train_(Times,ith_C3) = toc;

                            tic         
                            [tbsvm_struct_1] = ftbsvmtrain(Samples_Train,Labels_Train,Best_Parameter_1);
                            t_Train_1(Times,ith_C3) = toc;                            
                            
                            
                           % Predict the data
                            [Accp]= tbsvmclass(tbsvm_struct,Samples_Predict,Labels_Predict); 
                            C3_ACCp(Times,ith_C3) = Accp;
                            [Accp_]= tbsvmclass(tbsvm_struct_,Samples_Predict,Labels_Predict);   
                            C3_ACCp_(Times,ith_C3) = Accp_;
                            [Accp_1]= tbsvmclass(tbsvm_struct_1,Samples_Predict,Labels_Predict);
                            C3_ACCp_1(Times,ith_C3) = Accp_1;
                    end
                    
                    [Best_Acc,i]=max(C3_ACC(Times,:));
                    [Best_Acc_,j]=max(C3_ACC_(Times,:));
                    [Best_Acc_1,k]=max(C3_ACC_1(Times,:));
                    
                    Acc_Predict(Times) = C3_ACCp(Times,i);
                    Acc_Predict_(Times) = C3_ACCp_(Times,j);
                    Acc_Predict_1(Times) = C3_ACCp_1(Times,k);
                    
                    t_Train_Predict(Times) = t_Train(Times,i);
                    t_Train_Predict_(Times) = t_Train(Times,j);
                    t_Train_Predict_1(Times) = t_Train(Times,k);
                    
                    if Acc_Predict(Times)>Acc_Leader
                        Acc_Leader = Acc_Predict(Times);
                        Times_Leader = Times;
                        i_Leader = i;

                    end
                    
                    if Acc_Predict_(Times)>Acc_Leader_
                        Acc_Leader_ = Acc_Predict_(Times);
                        Times_Leader_ = Times;
                        j_Leader = j;

                    end
                    
                    if Acc_Predict_1(Times)>Acc_Leader_1
                        Acc_Leader_1 = Acc_Predict_1(Times);
                        Times_Leader_1 = Times;
                        k_Leader = k;

                    end
                end


            
                temp = [mean(t_Train_Predict),mean(Acc_Predict),std(Acc_Predict),Acc_Leader,...
                   Gam(Times_Leader,i_Leader),C3_C1(Times_Leader,i_Leader),C3_C3(Times_Leader,i_Leader)];
                xlswrite('result.xlsx',temp,['AC',num2str(iData+2),':AI',num2str(iData+2)]);
                
                xlswrite('result.xlsx',(C3_ACCp(1,:)),['C',num2str(iData+170),':O',num2str(iData+170)]);
                xlswrite('result.xlsx',(C3_ACCp(2,:)),['Q',num2str(iData+170),':AC',num2str(iData+170)]);
                xlswrite('result.xlsx',(C3_ACCp(3,:)),['AE',num2str(iData+170),':AQ',num2str(iData+170)]);
                xlswrite('result.xlsx',(C3_ACCp(4,:)),['AS',num2str(iData+170),':BE',num2str(iData+170)]);
                xlswrite('result.xlsx',(C3_ACCp(5,:)),['BG',num2str(iData+170),':BS',num2str(iData+170)]);
                xlswrite('result.xlsx',mean(C3_ACCp),['BU',num2str(iData+170),':CG',num2str(iData+170)]);
                
                xlswrite('result.xlsx',Acc_Predict',['CI',num2str(iData+170),':CM',num2str(iData+170)]);  %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的              
                
                
                temp_ = [mean(t_Train_Predict_),mean(Acc_Predict_),std(Acc_Predict_),Acc_Leader_,...
                   Gam_(Times_Leader_,j_Leader),C3_C1_(Times_Leader_,j_Leader),C3_C3_(Times_Leader_,j_Leader)];
                xlswrite('result.xlsx',temp_,['M',num2str(iData+2),':S',num2str(iData+2)]);  
                
                xlswrite('result.xlsx',(C3_ACCp_(1,:)),['C',num2str(iData+86),':O',num2str(iData+86)]);
                xlswrite('result.xlsx',(C3_ACCp_(2,:)),['Q',num2str(iData+86),':AC',num2str(iData+86)]);
                xlswrite('result.xlsx',(C3_ACCp_(3,:)),['AE',num2str(iData+86),':AQ',num2str(iData+86)]);
                xlswrite('result.xlsx',(C3_ACCp_(4,:)),['AS',num2str(iData+86),':BE',num2str(iData+86)]);
                xlswrite('result.xlsx',(C3_ACCp_(5,:)),['BG',num2str(iData+86),':BS',num2str(iData+86)]);
                xlswrite('result.xlsx',mean(C3_ACCp_),['BU',num2str(iData+86),':CG',num2str(iData+86)]);
                
                xlswrite('result.xlsx',Acc_Predict_',['CI',num2str(iData+86),':CM',num2str(iData+86)]);    %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的
                
                temp_1 = [mean(t_Train_Predict_1),mean(Acc_Predict_1),std(Acc_Predict_1),Acc_Leader_1,...
                   Gam_1(Times_Leader_1,k_Leader),C3_C1_1(Times_Leader_1,k_Leader),C3_C3_1(Times_Leader_1,k_Leader)];
                xlswrite('result.xlsx',temp_1,['U',num2str(iData+2),':AA',num2str(iData+2)]);  
                
                xlswrite('result.xlsx',(C3_ACCp_1(1,:)),['C',num2str(iData+128),':O',num2str(iData+128)]);
                xlswrite('result.xlsx',(C3_ACCp_1(2,:)),['Q',num2str(iData+128),':AC',num2str(iData+128)]);
                xlswrite('result.xlsx',(C3_ACCp_1(3,:)),['AE',num2str(iData+128),':AQ',num2str(iData+128)]);
                xlswrite('result.xlsx',(C3_ACCp_1(4,:)),['AS',num2str(iData+128),':BE',num2str(iData+128)]);
                xlswrite('result.xlsx',(C3_ACCp_1(5,:)),['BG',num2str(iData+128),':BS',num2str(iData+128)]);
                xlswrite('result.xlsx',mean(C3_ACCp_1),['BU',num2str(iData+128),':CG',num2str(iData+128)]);
                
                xlswrite('result.xlsx',Acc_Predict_1',['CI',num2str(iData+128),':CM',num2str(iData+128)]);    %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的                
                
                
            case 3   %%TSVM and FTSVM
            %% Train and predict the data
            

                %%%%%%%-----'------------Training the best parameters-----------------%%%%%%%
                t_Train = zeros(N_Times,length(C1_Interval));
                t_Train_ = zeros(N_Times,length(C1_Interval));
                Acc_Predict = zeros(N_Times, 1);
                Acc_Predict_ = zeros(N_Times, 1);
                t_Train_Predict = zeros(N_Times, 1);
                t_Train_Predict_ = zeros(N_Times, 1);
                
%                 MarginMEAN_Train = zeros(N_Times, 1);
%                 MarginSTD_Train = zeros(N_Times, 1);
%                 Accuracy = zeros();

                C3_ACC = zeros(N_Times,length(C1_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_ACCp = zeros(N_Times,length(C1_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_C1 = zeros(N_Times,length(C1_Interval));
                Gam = zeros(N_Times,length(gamma_Interval));

                C3_ACC_ = zeros(N_Times,length(C1_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_ACCp_ = zeros(N_Times,length(C1_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_C1_ = zeros(N_Times,length(C1_Interval));
                Gam_ = zeros(N_Times,length(gamma_Interval));

                Acc_Leader = 0;
                Acc_Leader_ = 0; 
                Times_Leader = 0;
                Times_Leader_ = 0;
                i_Leader = 0;
                j_Leader = 0;
                for Times = 1: N_Times
                    
                    [test, train] = Data_Rate(X, TrainRate);        
                    Ctrain=train(:,1:end-1);
                    dtrain= train(:,end);
                    Ctest= test(:,1:end-1);
                    dtest= test(:,end);
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



                      for ith_C1 = 1:length(C1_Interval) 
                           C1 = C1_Interval(ith_C1);    % lambda3
                           C2 = C1;
                           
                           Best_Acc = 0;%必须要清0，设为0后，其余参数就不需要再重置了。
                           Best_Acc_ = 0;

                                 for ith_gamma = 1:length(gamma_Interval)       %   gamma
                                     
                                    Indices = crossvalind('Kfold', length(Labels_Train),K_fold);
                                    Acc_SubPredict = zeros(K_fold, 1);
                                    Acc_SubPredict_ = zeros(K_fold, 1);                                     
                                   for repeat = 1:K_fold

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
                                        
                                        I_SubTrain = ~(Indices == repeat);
                                        Samples_SubTrain = Samples_Train(I_SubTrain,:);
                                        Labels_SubTrain = Labels_Train(I_SubTrain,:);

                                        I_SubPredict = ~I_SubTrain;
                                        Samples_SubPredict = Samples_Train(I_SubPredict,:);
                                        Labels_SubPredict = Labels_Train(I_SubPredict,:);


                                        I_SubA = Labels_SubTrain == 1;
                                        Samples_SubA = Samples_SubTrain(I_SubA,:);
                                        Labels_SubA = Labels_SubTrain(I_SubA);

                                        I_SubB = Labels_SubTrain == -1;
                                        Samples_SubB = Samples_SubTrain(I_SubB,:);                        
                                        Labels_SubB = Labels_SubTrain(I_SubB);     

                                        C_s.C1 = C1;
                                        C_s.C2 = C2;
%                                         s = IFuzzy_MemberShip(Samples_Train, Labels_Train, Kernel, Best_u);
%     %                                                     C_s.C3 = C3;
%     %                                                     C_s.C4 = C4;
%                                         C_s.s1 = s.s1;
%                                         C_s.s2 = s.s2;
                                        
                                        C_s_.C1 = C1;
                                        C_s_.C2 = C2;
                                        s_ = Fuzzy_MemberShip_(Samples_SubTrain, Labels_SubTrain, Kernel, Best_u);
                                        
                                        C_s_.s1 = s_(Labels_SubTrain==1);
                                        C_s_.s2 = s_ (Labels_SubTrain==-1);



                                        Subtbsvm_struct = Train_TSVM(Samples_SubA, Labels_SubA, Samples_SubB,Labels_SubB, Samples_SubTrain,C_s, Kernel, QPPs_Solver);
                                        Subtbsvm_struct_ = Train_FTSVM(Samples_SubA, Labels_SubA, Samples_SubB,Labels_SubB, Samples_SubTrain,C_s_, Kernel, QPPs_Solver);


                                        SubAcc = Predict_TSVM(Subtbsvm_struct, Samples_SubPredict,Labels_SubPredict, Samples_SubTrain);   
                                        SubAcc_ = Predict_FTSVM(Subtbsvm_struct_, Samples_SubPredict,Labels_SubPredict, Samples_SubTrain);


                                        Stop_Num = Stop_Num - 1;
                                        Acc_SubPredict(repeat) = SubAcc;
                                        Acc_SubPredict_(repeat) = SubAcc_;
                                        

                                        disp([num2str(Stop_Num), ' step(s) remaining.'])
                                   end


                                        if mean(Acc_SubPredict)>Best_Acc
                                            Best_Acc = mean(Acc_SubPredict);
                                            Best_C1 = C1;                       
                                            Best_Kernel = Kernel;
                                        end

                                        if mean(Acc_SubPredict_)>Best_Acc_
                                            Best_Acc_ = mean(Acc_SubPredict_);
                                            Best_C1_ = C1;                     
                                            Best_Kernel_ = Kernel;
                                        end

                                 end  % gamma
                                    
                            C3_ACC(Times,ith_C1) = Best_Acc;
                            C3_C1(Times,ith_C1) = Best_C1;
                            Gam(Times,ith_C1) = Best_Kernel.gamma;

                            C3_ACC_(Times,ith_C1) = Best_Acc_;
                            C3_C1_(Times,ith_C1) = Best_C1_;
                            Gam_(Times,ith_C1) = Best_Kernel_.gamma;
                            
                            
                            BestC_s.C1 = C3_C1(Times,ith_C1);
                            BestC_s.C2 = C3_C1(Times,ith_C1);
                            Best_Kernel.gamma = Gam(Times,ith_C1);
                            Best_Kernel.Type = 'RBF';
        %                     Bests = IFuzzy_MemberShip(Samples_Train, Labels_Train, Best_Kernel, Best_u); 
        %                     BestC_s.s1 = Bests.s1;
        %                     BestC_s.s2 = Bests.s2;

                            BestC_s_.C1 = C3_C1_(Times,ith_C1);
                            BestC_s_.C2 = C3_C1_(Times,ith_C1);
                            Best_Kernel_.gamma = Gam_(Times,ith_C1);
                            Best_Kernel_.Type = 'RBF';
                            Bests_ = Fuzzy_MemberShip_(Samples_Train, Labels_Train, Best_Kernel_, Best_u); 
                            BestC_s_.s1 = Bests_(Labels_Train == 1);
                            BestC_s_.s2 = Bests_(Labels_Train == -1);

                            I_A = Labels_Train == 1;
                            Samples_A = Samples_Train(I_A,:);
                            Labels_A = Labels_Train(I_A,:);

                            I_B = Labels_Train == -1;
                            Samples_B = Samples_Train(I_B,:);
                            Labels_B = Labels_Train(I_B,:);                    




                            tic         
                            tbsvm_struct = Train_TSVM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, BestC_s, Best_Kernel, QPPs_Solver);
                            t_Train(Times,ith_C1) = toc;

                            tic         
                            tbsvm_struct_ = Train_FTSVM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, BestC_s_, Best_Kernel_, QPPs_Solver);
                            t_Train_(Times,ith_C1) = toc;

                           % Predict the data
                            Accp = Predict_TSVM(tbsvm_struct, Samples_Predict,Labels_Predict, Samples_Train);   
                            Accp_ = Predict_FTSVM(tbsvm_struct_, Samples_Predict,Labels_Predict, Samples_Train);     
                            
                            C3_ACCp(Times,ith_C1) = Accp;
                            C3_ACCp_(Times,ith_C1) = Accp_;
                                                                        
                      end
                      
                      
                    [Best_Acc,i]=max(C3_ACC(Times,:));
                    [Best_Acc_,j]=max(C3_ACC_(Times,:));
                    
                    Acc_Predict(Times)  = C3_ACCp(Times,i);
                    Acc_Predict_(Times)  = C3_ACCp_(Times,j);
                    t_Train_Predict(Times) = t_Train(Times,i);
                    t_Train_Predict_(Times) = t_Train_(Times,j);
                    
                    
                    
                    if Acc_Predict(Times)>Acc_Leader
                        Acc_Leader = Acc_Predict(Times);
                        Times_Leader = Times;
                        i_Leader = i;

                    end                    
                    
                    if Acc_Predict_(Times)>Acc_Leader_
                        Acc_Leader_ = Acc_Predict_(Times);
                        Times_Leader_ = Times;
                        j_Leader = j;

                    end  

                end

             

                temp = [mean(t_Train_Predict),100*mean(Acc_Predict),100*std(Acc_Predict),100*Acc_Leader,...
                   Gam(Times_Leader,i_Leader),C3_C1(Times_Leader,i_Leader)];
                xlswrite('result.xlsx',temp,['AK',num2str(iData+2),':AP',num2str(iData+2)]);
                
                xlswrite('result.xlsx',100*(C3_ACCp(1,:)),['C',num2str(iData+212),':O',num2str(iData+212)]);
                xlswrite('result.xlsx',100*(C3_ACCp(2,:)),['Q',num2str(iData+212),':AC',num2str(iData+212)]);
                xlswrite('result.xlsx',100*(C3_ACCp(3,:)),['AE',num2str(iData+212),':AQ',num2str(iData+212)]);
                xlswrite('result.xlsx',100*(C3_ACCp(4,:)),['AS',num2str(iData+212),':BE',num2str(iData+212)]);
                xlswrite('result.xlsx',100*(C3_ACCp(5,:)),['BG',num2str(iData+212),':BS',num2str(iData+212)]);
                xlswrite('result.xlsx',100*mean(C3_ACCp),['BU',num2str(iData+212),':CG',num2str(iData+212)]);
                
                xlswrite('result.xlsx',100*Acc_Predict',['CI',num2str(iData+212),':CM',num2str(iData+212)]);  %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的                

                temp_ = [mean(t_Train_Predict_),100*mean(Acc_Predict_),100*std(Acc_Predict_),100*Acc_Leader_,...
                   Gam(Times_Leader_,j_Leader),C3_C1(Times_Leader_,j_Leader)];
                xlswrite('result.xlsx',temp_,['AR',num2str(iData+2),':AW',num2str(iData+2)]);
                
                xlswrite('result.xlsx',100*(C3_ACCp_(1,:)),['C',num2str(iData+254),':O',num2str(iData+254)]);
                xlswrite('result.xlsx',100*(C3_ACCp_(2,:)),['Q',num2str(iData+254),':AC',num2str(iData+254)]);
                xlswrite('result.xlsx',100*(C3_ACCp_(3,:)),['AE',num2str(iData+254),':AQ',num2str(iData+254)]);
                xlswrite('result.xlsx',100*(C3_ACCp_(4,:)),['AS',num2str(iData+254),':BE',num2str(iData+254)]);
                xlswrite('result.xlsx',100*(C3_ACCp_(5,:)),['BG',num2str(iData+254),':BS',num2str(iData+254)]);
                xlswrite('result.xlsx',100*mean(C3_ACCp_),['BU',num2str(iData+254),':CG',num2str(iData+254)]);
                
                xlswrite('result.xlsx',100*Acc_Predict_',['CI',num2str(iData+254),':CM',num2str(iData+254)]);  %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的                
                
           case 4   %%SVM and FSVM
            %% Train and predict the data

                %%%%%%%-----------------Training the best parameters-----------------%%%%%%%
                t_Train = zeros(N_Times,length(C1_Interval));
                t_Train_ = zeros(N_Times,length(C1_Interval));
                Acc_Predict = zeros(N_Times, 1);
                Acc_Predict_ = zeros(N_Times, 1);
%                 MarginMEAN_Train = zeros(N_Times, 1);
%                 MarginSTD_Train = zeros(N_Times, 1);
%                 Accuracy = zeros();
               t_Train_Predict = zeros(N_Times,1);
               t_Train_Predict_ = zeros(N_Times,1);
               
                C3_ACC = zeros(N_Times,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_ACCp = zeros(N_Times,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_C1 = zeros(N_Times,length(C1_Interval));
                Gam = zeros(N_Times,length(gamma_Interval));

                C3_ACC_ = zeros(N_Times,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_ACCp_ = zeros(N_Times,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_C1_ = zeros(N_Times, length(C1_Interval));
                Gam_ = zeros(N_Times, length(gamma_Interval));               
               
                Acc_Leader = 0;
                Acc_Leader_ = 0;
                Times_Leader = 0;
                Times_Leader_ = 0;                
                i_Leader = 0;
                j_Leader = 0;
                n = 0;
                for Times = 1: N_Times
                    
                    [test, train] = Data_Rate(X, TrainRate);
                    
                    

                    
                    Ctrain=train(:,1:end-1);
                    dtrain= train(:,end);
                    Ctest= test(:,1:end-1);
                    dtest= test(:,end);
                    

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
 
                    
                    




                      for ith_C1 = 1:length(C1_Interval) 
                           C1 = C1_Interval(ith_C1);    % lambda3
                           
                           Best_Acc = 0;%必须要清0，设为0后，其余参数就不需要再重置了。
                           Best_Acc_ = 0;

                                 for ith_gamma = 1:length(gamma_Interval)       %   gamma

                                        Acc_SubPredict = zeros(K_fold, 1);
                                        Acc_SubPredict_ = zeros(K_fold, 1);
                                        Indices = crossvalind('Kfold', length(Labels_Train),K_fold);
                                        
                                     for repeat = 1:K_fold
                                        


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
                                        
                                        I_SubTrain = ~(Indices == repeat);
                                        Samples_SubTrain = Samples_Train(I_SubTrain,:);
                                        Labels_SubTrain = Labels_Train(I_SubTrain,:);

                                        I_SubPredict = ~I_SubTrain;
                                        Samples_SubPredict = Samples_Train(I_SubPredict,:);
                                        Labels_SubPredict = Labels_Train(I_SubPredict,:);

                                        
%                                         s = Fuzzy_MemberShip(Samples_Train,Labels_Train, Kernel, Best_u);
                                        s_ = Fuzzy_MemberShip_(Samples_SubTrain,Labels_SubTrain, Kernel, Best_u);


                                        Outs_SubTrain =  Train_SVM(Samples_SubTrain, Labels_SubTrain, C1*abs(Labels_SubTrain), Kernel, QPPs_Solver);
                                        Outs_SubTrain_ =  Train_SVM(Samples_SubTrain, Labels_SubTrain, C1*s_, Kernel, QPPs_Solver);

                                        SubAcc = Predict_SVM(Outs_SubTrain, Samples_SubPredict, Labels_SubPredict);  
                                        SubAcc_ = Predict_SVM(Outs_SubTrain_, Samples_SubPredict, Labels_SubPredict); 


                                        Stop_Num = Stop_Num - 1;
                                        Acc_SubPredict(repeat) = SubAcc;
                                        Acc_SubPredict_(repeat) = SubAcc_;                                        

                                        disp([num2str(Stop_Num), ' step(s) remaining.'])
                                     end

                                        if mean(Acc_SubPredict)>Best_Acc
                                            Best_Acc = mean(Acc_SubPredict);
                                            Best_C1 = C1;                     
                                            Best_Kernel = Kernel;
                                        end
                                        
                                        if mean(Acc_SubPredict_)>Best_Acc_
                                            Best_Acc_ = mean(Acc_SubPredict_);
                                            Best_C1_ = C1;                     
                                            Best_Kernel_ = Kernel;
                                        end


                                end  % gamma
                                    C3_ACC(Times,ith_C1) = Best_Acc;
                                    C3_C1(Times,ith_C1) = Best_C1;
                                    Gam(Times,ith_C1) = Best_Kernel.gamma;

                                    C3_ACC_(Times,ith_C1) = Best_Acc_;
                                    C3_C1_(Times,ith_C1) = Best_C1_;
                                    Gam_(Times,ith_C1) = Best_Kernel_.gamma;
                                    

                                    
                                    Best_C1 = C3_C1(Times,ith_C1);
                                    Best_Kernel.gamma = Gam(Times,ith_C1);
                %                     Bests = Fuzzy_MemberShip_(Samples_Train, Labels_Train, Best_Kernel, Best_u); 



                                    Best_C1_ = C3_C1_(Times,ith_C1);
                                    Best_Kernel_.gamma = Gam_(Times,ith_C1);
                                    Bests_ = Fuzzy_MemberShip_(Samples_Train, Labels_Train, Best_Kernel, Best_u); 


                                    tic         
                                    Outs_Train =  Train_SVM(Samples_Train, Labels_Train, Best_C1*abs(Labels_Train), Best_Kernel, QPPs_Solver);
                                    t_Train(Times,ith_C1) = toc;

                                    tic         
                                    Outs_Train_ =  Train_SVM(Samples_Train, Labels_Train, Best_C1_*Bests_, Best_Kernel_, QPPs_Solver);
                                    t_Train_(Times,ith_C1) = toc;

                                   % Predict the data
                                    Accp = Predict_SVM(Outs_Train, Samples_Predict, Labels_Predict);  
                                    Accp_ = Predict_SVM(Outs_Train_, Samples_Predict, Labels_Predict);  
                                    
                                    C3_ACCp(Times,ith_C1) = Accp;
                                    C3_ACCp_(Times,ith_C1) = Accp_;
                                    
                      end
                      
                    [Best_Acc,i]=max(C3_ACC(Times,:));
                    [Best_Acc_,j]=max(C3_ACC_(Times,:));
                    
                    Best_C1 = C3_C1(i);
                    Best_Kernel.gamma = Gam(i);
%                     Bests = Fuzzy_MemberShip_(Samples_Train, Labels_Train, Best_Kernel, Best_u); 

                    

                    Best_C1_ = C3_C1_(j);
                    Best_Kernel_.gamma = Gam_(j);
                    Bests_ = Fuzzy_MemberShip_(Samples_Train, Labels_Train, Best_Kernel, Best_u); 
                    
                    Acc_Predict(Times)  = C3_ACCp(Times,i);
                    Acc_Predict_(Times)  = C3_ACCp_(Times,j);
                    t_Train_Predict(Times) = t_Train(Times,i);
                    t_Train_Predict_(Times) = t_Train_(Times,j);                    
                    
                    if Acc_Predict(Times)>Acc_Leader
                        Acc_Leader = Acc_Predict(Times);
                        Times_Leader = Times;
                        i_Leader = i;

                    end    
                    
                    if Acc_Predict_(Times)>Acc_Leader_
                        Acc_Leader_ = Acc_Predict_(Times);
                        Times_Leader_ = Times;
                        j_Leader = j;

                    end                      


                end


%             
                temp = [mean(t_Train_Predict),100*mean(Acc_Predict),100*std(Acc_Predict),100*Acc_Leader,...
                   Gam(Times,i_Leader),C3_C1(Times,i_Leader)];
                xlswrite('result.xlsx',temp,['AY',num2str(iData+2),':BD',num2str(iData+2)]);
                
                xlswrite('result.xlsx',100*(C3_ACCp(1,:)),['C',num2str(iData+296),':O',num2str(iData+296)]);
                xlswrite('result.xlsx',100*(C3_ACCp(2,:)),['Q',num2str(iData+296),':AC',num2str(iData+296)]);
                xlswrite('result.xlsx',100*(C3_ACCp(3,:)),['AE',num2str(iData+296),':AQ',num2str(iData+296)]);
                xlswrite('result.xlsx',100*(C3_ACCp(4,:)),['AS',num2str(iData+296),':BE',num2str(iData+296)]);
                xlswrite('result.xlsx',100*(C3_ACCp(5,:)),['BG',num2str(iData+296),':BS',num2str(iData+296)]);
                xlswrite('result.xlsx',100*mean(C3_ACCp),['BU',num2str(iData+296),':CG',num2str(iData+296)]);
                
                xlswrite('result.xlsx',100*Acc_Predict',['CI',num2str(iData+296),':CM',num2str(iData+296)]);  %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的                
                
                temp = [mean(t_Train_Predict_),100*mean(Acc_Predict_),100*std(Acc_Predict_),100*Acc_Leader_,...
                   Gam_(Times,j_Leader),C3_C1_(Times,j_Leader)];
                xlswrite('result.xlsx',temp,['BF',num2str(iData+2),':BK',num2str(iData+2)]);
                
                xlswrite('result.xlsx',100*(C3_ACCp_(1,:)),['C',num2str(iData+338),':O',num2str(iData+338)]);
                xlswrite('result.xlsx',100*(C3_ACCp_(2,:)),['Q',num2str(iData+338),':AC',num2str(iData+338)]);
                xlswrite('result.xlsx',100*(C3_ACCp_(3,:)),['AE',num2str(iData+338),':AQ',num2str(iData+338)]);
                xlswrite('result.xlsx',100*(C3_ACCp_(4,:)),['AS',num2str(iData+338),':BE',num2str(iData+338)]);
                xlswrite('result.xlsx',100*(C3_ACCp_(5,:)),['BG',num2str(iData+338),':BS',num2str(iData+338)]);
                xlswrite('result.xlsx',100*mean(C3_ACCp_),['BU',num2str(iData+338),':CG',num2str(iData+338)]);
                
                xlswrite('result.xlsx',100*Acc_Predict_',['CI',num2str(iData+338),':CM',num2str(iData+338)]);    %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的             
                
            case 5  %%LDM and FLDM
            %% Train and predict the data
            

            

                %%%%%%%-----------------Training the best parameters-----------------%%%%%%%
                t_Train = zeros(N_Times,length(C1_Interval));
                t_Train_ = zeros(N_Times, length(C1_Interval));
                Acc_Predict = zeros(N_Times,1);
                Acc_Predict_ = zeros(N_Times,1);
                
                t_Train_Predict = zeros(N_Times,1);
                t_Train_Predict_ = zeros(N_Times,1);
%                 MarginMEAN_Train = zeros(Times,N_Times);
%                 MarginSTD_Train = zeros(Times,N_Times);
%                 Accuracy = zeros();

                C3_ACC = zeros(N_Times,length(C1_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_ACCp = zeros(N_Times,length(C1_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_lambda1 = zeros(N_Times,length(lambda1_Interval));
                C3_lambda2 = zeros(N_Times,length(lambda2_Interval));
                C3_C1 = zeros(N_Times,length(C1_Interval));
                Gam = zeros(N_Times,length(gamma_Interval));

                C3_ACC_ = zeros(N_Times,length(C1_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_ACCp_ = zeros(N_Times,length(C1_Interval));%由于xlwrite是写行向量，所以设置为行向量
                C3_lambda1_ = zeros(N_Times,length(lambda1_Interval));
                C3_lambda2_ = zeros(N_Times,length(lambda2_Interval));
                C3_C1_ = zeros(N_Times,length(C1_Interval));
                Gam_ = zeros(N_Times,length(gamma_Interval));
                
                Times_Leader = 0;
                Times_Leader_ = 0;
                i_Leader = 0;
                j_Leader = 0;
                
                Acc_Leader = 0;
                Acc_Leader_ = 0;
                for Times = 1: N_Times

                    [test, train] = Data_Rate(X, TrainRate);        
                    Ctrain=train(:,1:end-1);
                    dtrain= train(:,end);
                    Ctest= test(:,1:end-1);
                    dtest= test(:,end);                    
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
                    


                      for ith_C1 = 1:length(C1_Interval)                           
                          C1 = C1_Interval(ith_C1);    % lambda3
                          
                          Best_Acc = 0;%必须要清0，设为0后，其余参数就不需要再重置了。
                           Best_Acc_ = 0;
                          
                            for ith_lambda1 = 1:length(lambda1_Interval)    % lambda1
                                lambda1 = lambda1_Interval(ith_lambda1);    % lambda1
                                for ith_lambda2 = 1:length(lambda2_Interval)
                                  lambda2 = lambda2_Interval(ith_lambda2);



                                            for ith_gamma = 1:length(gamma_Interval)       %   gamma

                                                        Acc_SubPredict = zeros(K_fold, 1);
                                                        Acc_SubPredict_ = zeros(K_fold, 1);                                                         
                                                        Indices = crossvalind('Kfold', length(Labels_Train),K_fold);

                                                     for repeat = 1:K_fold

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
                                                            
                                                            I_SubTrain = ~(Indices == repeat);
                                                            Samples_SubTrain = Samples_Train(I_SubTrain,:);
                                                            Labels_SubTrain = Labels_Train(I_SubTrain,:);

                                                            I_SubPredict = ~I_SubTrain;
                                                            Samples_SubPredict = Samples_Train(I_SubPredict,:);
                                                            Labels_SubPredict = Labels_Train(I_SubPredict,:);
                                                            
                                                            C_s_.C = C1*abs(Labels_SubTrain); %%!!!!
                                                            C_s_.s = Fuzzy_MemberShip_(Samples_SubTrain, Labels_SubTrain,Kernel,Best_u);


                                                            Outs_SubTrain = Train_LDM(Samples_SubTrain, Labels_SubTrain, lambda1, lambda2, C1*abs(Labels_SubTrain),Kernel, QPPs_Solver);
                                                            Outs_SubTrain_ = Train_FLDM(Samples_SubTrain, Labels_SubTrain, lambda1, lambda2, C_s_, FLDM_Type, Kernel, QPPs_Solver);

                                                            SubAcc = Predict_LDM(Outs_SubTrain, Samples_SubPredict, Labels_SubPredict);  
                                                            SubAcc_ = Predict_FLDM(Outs_SubTrain_, Samples_SubPredict, Labels_SubPredict);


                                                            Stop_Num = Stop_Num - 1;
                                                            Acc_SubPredict(repeat) = SubAcc;
                                                            Acc_SubPredict_(repeat) = SubAcc_; 
                                                            disp([num2str(Stop_Num), ' step(s) remaining.'])
                                                     end

                                                            if mean(Acc_SubPredict)>Best_Acc
                                                                Best_Acc = mean(Acc_SubPredict);
                                                                Best_lambda1 = lambda1;
                                                                Best_lambda2 = lambda2;
                                                                Best_C1 = C1;                     
                                                                Best_Kernel = Kernel;
                                                            end
                                                            
                                                             if mean(Acc_SubPredict_)>Best_Acc_
                                                                Best_Acc_ = mean(Acc_SubPredict_);
                                                                Best_lambda1_ = lambda1;
                                                                Best_lambda2_ = lambda2;
                                                                Best_C1_ = C1;                     
                                                                Best_Kernel_ = Kernel;
                                                            end

                                           end  % gamma

                                end
                            end    % lambda1
                            C3_ACC(Times,ith_C1) = Best_Acc;
                            C3_lambda1(Times,ith_C1) = Best_lambda1;
                            C3_lambda2(Times,ith_C1) = Best_lambda2;
                            C3_C1(Times,ith_C1) = Best_C1;
                            Gam(Times,ith_C1) = Best_Kernel.gamma;
                            
                            C3_ACC_(Times,ith_C1) = Best_Acc_;
                            C3_lambda1_(Times,ith_C1) = Best_lambda1_;
                            C3_lambda2_(Times,ith_C1) = Best_lambda2_;
                            C3_C1_(Times,ith_C1) = Best_C1_;
                            Gam_(Times,ith_C1) = Best_Kernel_.gamma;
                            
                            
                            Best_C1 = C3_C1(Times,ith_C1);
                            Best_lambda1 = C3_lambda1(Times,ith_C1);
                            Best_lambda2 = C3_lambda2(Times,ith_C1);
                            Best_Kernel.gamma = Gam(Times,ith_C1);
    %                         BestC_s.C = Best_C1*abs(Labels_Train);
    %                         BestC_s.s = IFuzzy_MemberShip_wang19(Samples_Train, Labels_Train,Best_Kernel,Best_u);

                            Best_C1_ = C3_C1_(Times,ith_C1);
                            Best_lambda1_ = C3_lambda1_(Times,ith_C1);
                            Best_lambda2_ = C3_lambda2_(Times,ith_C1);
                            Best_Kernel_.gamma = Gam_(Times,ith_C1);
                            BestC_s_.C = Best_C1_*abs(Labels_Train);
                            BestC_s_.s = Fuzzy_MemberShip_(Samples_Train, Labels_Train,Best_Kernel_,Best_u);

                            tic         
                            Outs_Train = Train_LDM(Samples_Train, Labels_Train, Best_lambda1, Best_lambda2, Best_C1*abs(Labels_Train), Best_Kernel, QPPs_Solver);
                            t_Train(Times,ith_C1) = toc;

                            tic         
                            Outs_Train_ = Train_FLDM(Samples_Train, Labels_Train, Best_lambda1_, Best_lambda2_, BestC_s_, FLDM_Type, Best_Kernel_, QPPs_Solver);
                            t_Train_(Times,ith_C1) = toc;


                           % Predict the data
                            Accp = Predict_LDM(Outs_Train, Samples_Predict, Labels_Predict);  
                            Accp_ = Predict_FLDM(Outs_Train_, Samples_Predict, Labels_Predict);  
                            
                            C3_ACCp(Times,ith_C1) = Accp;
                            C3_ACCp_(Times,ith_C1) = Accp_;
                            
                      end
                      [Best_Acc,i]=max(C3_ACC(Times,:));
                      [Best_Acc_,j]=max(C3_ACC_(Times,:));
                    
                        Acc_Predict(Times)  = C3_ACCp(Times,i);
                        Acc_Predict_(Times)  = C3_ACCp_(Times,j);
                        t_Train_Predict(Times) = t_Train(Times,i);
                        t_Train_Predict_(Times) = t_Train_(Times,j);                    

                        if Acc_Predict(Times)>Acc_Leader
                            Acc_Leader = Acc_Predict(Times);
                            Times_Leader = Times;
                            i_Leader = i;

                        end    

                        if Acc_Predict_(Times)>Acc_Leader_
                            Acc_Leader_ = Acc_Predict_(Times);
                            Times_Leader_ = Times;
                            j_Leader = j;

                        end


                end


                %%%%%%%---------------------Save the statiatics---------------------%%%%%%%
%                 Name = Str_Name(10:end-4);
                %                 Loc_Nam = [Location, '\', Name, '.txt'];
                %                 f = fopen(Loc_Nam, 'wt');
                disp('LDM');
            
                temp = [mean(t_Train_Predict),100*mean(Acc_Predict),100*std(Acc_Predict),100*Acc_Leader,...
                   Gam(Times_Leader,i_Leader),C3_lambda1(Times_Leader,i_Leader),C3_lambda2(Times_Leader,i_Leader),C3_C1(Times_Leader,i_Leader)];
                xlswrite('result.xlsx',temp,['BM',num2str(iData+2),':BT',num2str(iData+2)]); 
                
                xlswrite('result.xlsx',100*(C3_ACCp(1,:)),['C',num2str(iData+381),':O',num2str(iData+381)]);
                xlswrite('result.xlsx',100*(C3_ACCp(2,:)),['Q',num2str(iData+381),':AC',num2str(iData+381)]);
                xlswrite('result.xlsx',100*(C3_ACCp(3,:)),['AE',num2str(iData+381),':AQ',num2str(iData+381)]);
                xlswrite('result.xlsx',100*(C3_ACCp(4,:)),['AS',num2str(iData+381),':BE',num2str(iData+381)]);
                xlswrite('result.xlsx',100*(C3_ACCp(5,:)),['BG',num2str(iData+381),':BS',num2str(iData+381)]);
                xlswrite('result.xlsx',100*mean(C3_ACCp),['BU',num2str(iData+381),':CG',num2str(iData+381)]);
                
                xlswrite('result.xlsx',100*Acc_Predict',['CI',num2str(iData+381),':CM',num2str(iData+381)]); %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的                   
                
                temp_ = [mean(t_Train_Predict_),100*mean(Acc_Predict_),100*std(Acc_Predict_),100*Acc_Leader_,...
                    Gam_(Times_Leader_,j_Leader),C3_lambda1_(Times_Leader_,j_Leader),C3_lambda2_(Times_Leader_,j_Leader),C3_C1_(Times_Leader_,j_Leader)];
                xlswrite('result.xlsx',temp_,['BV',num2str(iData+2),':CC',num2str(iData+2)]);
                
                xlswrite('result.xlsx',100*(C3_ACCp_(1,:)),['C',num2str(iData+422),':O',num2str(iData+422)]);
                xlswrite('result.xlsx',100*(C3_ACCp_(2,:)),['Q',num2str(iData+422),':AC',num2str(iData+422)]);
                xlswrite('result.xlsx',100*(C3_ACCp_(3,:)),['AE',num2str(iData+422),':AQ',num2str(iData+422)]);
                xlswrite('result.xlsx',100*(C3_ACCp_(4,:)),['AS',num2str(iData+422),':BE',num2str(iData+422)]);
                xlswrite('result.xlsx',100*(C3_ACCp_(5,:)),['BG',num2str(iData+422),':BS',num2str(iData+422)]);
                xlswrite('result.xlsx',100*mean(C3_ACCp_),['BU',num2str(iData+422),':CG',num2str(iData+422)]);
                  
                xlswrite('result.xlsx',100*Acc_Predict_',['CI',num2str(iData+422),':CM',num2str(iData+422)]);     %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的               
                
        end
    end
end   
