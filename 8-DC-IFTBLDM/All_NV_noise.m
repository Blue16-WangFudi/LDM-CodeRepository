%% Initilizing the enviroment
clear all
close all
clc

%% Data preparation
%     Str(1).Name = 'Data_mat\Australian.mat';
%     Str(2).Name = 'Data_mat\Breast_cancer(Original).mat';
%     Str(3).Name = 'Data_mat\Glass.mat';
%     Str(4).Name = 'Data_mat\Ripley_Predict.mat';
%     Str(5).Name = 'Data_mat_x\credit-a.mat';
%     Str(6).Name = 'Data_mat_x\credit-g.mat';
%     Str(7).Name = 'Data_mat_x\heart-statlog.mat';
%     Str(8).Name = 'Data_mat_x\hepatitis.mat';
%     Str(9).Name = 'Data_mat_x\tic-tac-toe.mat';
% 
%     Str(10).Name = 'Data_mat\CMC.mat';
%     Str(11).Name = 'Data_mat\Haberman.mat';
%     Str(12).Name = 'Data_mat\Ripley_Train';
%     Str(13).Name = 'Data_mat\Ionosphere.mat';
%     Str(14).Name = 'Data_mat_x\clean1.mat';
%     Str(15).Name = 'Data_mat_x\cmc_0_1.mat';
%     Str(16).Name = 'Data_mat_x\cmc_0_2.mat';

%     Str(17).Name = 'Data_mat_n\Mook_1.mat';
%     Str(18).Name = 'Data_mat_n\Mook_2.mat';
%     Str(19).Name = 'Data_mat_n\Mook_3.mat';
%     Str(20).Name = 'Data_mat_n\Ecoil.mat';
%     Str(21).Name = 'Data_mat_n\Fertility.mat';
%     Str(22).Name = 'Data_mat_n\Prlx.mat';
%     Str(23).Name = 'Data_mat_n\Spect.mat';
%     Str(24).Name = 'Data_mat_n\votes.mat';
%     Str(25).Name = 'Data_mat_n\Wdbc.mat';
%     Str(26).Name = 'Data_mat_n\Wpbc.mat';

%     Str(27).Name = 'Data_mat\Statlog_Heart.mat';
%     Str(28).Name = 'Data_mat\BUPA.mat';
%     Str(29).Name = 'Data_mat\German.mat';
%     Str(30).Name = 'Data_mat\Hepatitis.mat';
%     Str(31).Name = 'Data_mat\Iris.mat';
%    
%     Str(32).Name = 'Data_mat_x\breast-w.mat';
%     Str(33).Name = 'Data_mat_x\cmc_1_2.mat';
%     Str(34).Name = 'Data_mat_x\cylinder-bands.mat';
%     Str(35).Name = 'Data_mat_x\diabetes.mat';
%     Str(36).Name = 'Data_mat_x\Ionosphere.mat';
%     Str(37).Name = 'Data_mat\New_thyroid.mat';
%     Str(38).Name = 'Data_mat\Pima_indians.mat';
%     Str(39).Name = 'Data_mat\Promoters.mat';
%     Str(40).Name = 'Data_mat\Sonar.mat';
%     Str(41).Name = 'Data_mat\Wine.mat';


%%  Choose the Classifier
% case 1:FTBLDM and TBLDM; 2:TLDM; 3:TBSVM; 4:FTSVM and TSVM;
%% Some parameters
% F_LDM Type
FLDM_Type = 'F2_LDM';
Kernel.Type = 'RBF';
QPPs_Solver = 'QP_Matlab';
gamma_Interval = 2.^(-5: 2);
lambda1_Interval = 2.^(-8: 4);
lambda2_Interval = 2.^(-8: 4);
C1_Interval = 2.^(-8: 4);
C3_Interval = 2.^(-8: 4);


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


for iData = 58:63
%                 Str_Name = Str(iData).Name;
%                 Output = load(Str_Name);
%                 Data_Name = fieldnames(Output);   % A struct data
%                 Data_Original = getfield(Output, Data_Name{1}); % Abstract the data
%                 % Normalization
%                 Data_Original = [mapminmax(Data_Original(:, 1:end-1)', 0, 1)', Data_Original(:, end)]; % Map the original data to value between [0, 1] by colum
% %                 M_Original = size(Data_Original, 1);
% %                 Data_Original = Data_Original(randperm(M_Original), :);

    if (iData==43)
        load('Data_mat_n\monks_1_train.txt');
        load('Data_mat_n\monks_1_test.txt');
        Ctrain= monks_1_train(:,2:end);
        dtrain=  monks_1_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= monks_1_test(:,2:end);
        dtest=  monks_1_test(:,1);
        dtest(find(dtest==0))=-1;
    end

    if(iData==40)
        load('Data_mat_n\monks_2_train.txt');
        load('Data_mat_n\monks_2_test.txt');
        Ctrain= monks_2_train(:,2:end);
        dtrain=  monks_2_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= monks_2_test(:,2:end);
        dtest=  monks_2_test(:,1);
        dtest(find(dtest==0))=-1;
    end

    if (iData==44)
        load('Data_mat_n\monks_3_train.txt');
        load('Data_mat_n\monks_3_test.txt');
        Ctrain= monks_3_train(:,2:end);
        dtrain=  monks_3_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= monks_3_test(:,2:end);
        dtest=  monks_3_test(:,1);
        dtest(find(dtest==0))=-1;
    end

    if(iData==60)
        load('Data_mat_n\ecoli_data.mat');
        %         rng(2);
        X= ecoli_data(:,1:end-1);
        Y= ecoli_data(:,end);
        Y(find(Y~=1))=-1;
        r1=randperm(size(X,1));
        X = X(r1,:);
        Y=Y(r1,:);
        Ctrain=X(1:200,:);
        dtrain= Y(1:200,:);
        Ctest= X(201:end,:);
        dtest= Y(201:end,:);
    end

    if(iData==52)
        data=xlsread('Data_mat_n\fertility_Diagnosis.xlsx');
        X= data(:,1:end-1);
        Y= data(:,end);
        Y(find(Y==0))=-1;
        Ctrain=X(1:50,:);
        dtrain= Y(1:50,:);
        Ctest= X(51:end,:);
        dtest= Y(51:end,:);
    end

    if(iData==41)
        load('Data_mat_n\plrx.txt');
        X=plrx(:,1:end-1);
        Y= plrx(:,end);
        Y(find(Y==2))=-1;
        Ctrain=X(1:100,:);
        dtrain= Y(1:100,:);
        Ctest= X(101:end,:);
        dtest= Y(101:end,:);

    end
    if (iData==58)
        load('Data_mat_n\SPECT_train.txt')
        load('Data_mat_n\SPECT_test.txt');
        Ctrain = SPECT_train(:,2:end);
        dtrain=  SPECT_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= SPECT_test(:,2:end);
        dtest=  SPECT_test(:,1);
        dtest(find(dtest==0))=-1;
    end   
    if (iData==5)
        load('Data_mat_n\votes.mat')
        X=votes(:,2:end);
        Y= votes(:,1);
        Y(find(Y==2))=-1;
        Ctrain=X(1:200,:);
        dtrain= Y(1:200,:);
        Ctest= X(201:end,:);
        dtest= Y(201:end,:);
    end

    if (iData==53)
        load('Data_mat_n\wdbc_data.mat')
        load('Data_mat_n\wdbc_label.mat')
        X=wdbc_data;
        Y= wdbc_label;
        Y(find(Y==2))=-1;
        Ctrain=X(1:400,:);
        dtrain= Y(1:400,:);
        Ctest= X(401:end,:);
        dtest= Y(401:end,:);
    end

    if (iData==42)
        load('Data_mat_n\Wpbc.mat');
        X = Wpbc;
        Ctrain=X(1:90,1:end-1);
        dtrain= X(1:90,end);
        Ctest= X(91:end,1:end-1);
        dtest= X(91:end,end);

    end
    if (iData==54)
        load('Data_mat_n\echocardiogram_data.mat');
        load('Data_mat_n\echocardiogram_label.mat');
        X=x;
        Y= y;
        Y(find(Y==0))=-1;
        Ctrain=X(1:80,:);
        dtrain= Y(1:80,:);
        Ctest= X(81:end,:);
        dtest= Y(81:end,:);   
    end
    
     if (iData==49)
        load('Data_mat\Australian.mat');
        X = Australian;
        Ctrain=X(1:395,1:end-1);
        dtrain= X(1:395,end);
        Ctest= X(396:end,1:end-1);
        dtest= X(396:end,end);

     end
    
     if (iData==45)
        load('Data_mat_x\credit_a.mat');
        X = credit_a;        
        Ctrain=X(1:327,1:end-1);
        dtrain= X(1:327,end);
        Ctest= X(328:end,1:end-1);
        dtest= X(328:end,end);

     end
     
     if (iData==46)
        load('Data_mat_x\heart_statlog.mat');
        X = heart_statlog;
        Ctrain=X(1:135,1:end-1);
        dtrain= X(1:135,end);
        Ctest= X(136:end,1:end-1);
        dtest= X(136:end,end);

     end
     
     if (iData==48)
        load( 'Data_mat\Hepatitis.mat');
        X = Hepatitis;
        Ctrain=X(1:78,1:end-1);
        dtrain= X(1:78,end);
        Ctest= X(79:end,1:end-1);
        dtest= X(79:end,end);

     end
     
     if (iData==24)
        load('Data_mat\Ripley_Train.mat')
        load('Data_mat\Ripley_Predict.mat');
        Ctrain = Ripley_Train(:,1:end-1);
        dtrain=  Ripley_Train(:,end);
        Ctest= Ripley_Predict(:,1:end-1);
        dtest=  Ripley_Predict(:,end);  
    end
    
     if (iData==47)
        load( 'Data_mat\Glass.mat');
        X = Glass;
        Ctrain=X(1:107,1:end-1);
        dtrain= X(1:107,end);
        Ctest= X(108:end,1:end-1);
        dtest= X(108:end,end);

     end
    
      
     if (iData==26)
        load( 'Data_mat\CMC.mat');
        X = CMC;
        Ctrain=X(1:737,1:end-1);
        dtrain= X(1:737,end);
        Ctest= X(738:end,1:end-1);
        dtest= X(738:end,end);

     end
      
     if (iData==50)
        load( 'Data_mat\Haberman.mat');
        X = Haberman;
        Ctrain=X(1:153,1:end-1);
        dtrain= X(1:153,end);
        Ctest= X(154:end,1:end-1);
        dtest= X(154:end,end);
        
     end
      
     if (iData==55)
        load( 'Data_mat_x\tic_tac_toe.mat');
        X = tic_tac_toe;
        Ctrain=X(1:479,1:end-1);
        dtrain= X(1:479,end);
        Ctest= X(480:end,1:end-1);
        dtest= X(480:end,end);

     end
    
      
     if (iData==62)
        load( 'Data_mat\Breast_cancer_Original.mat');
        X = Breast_cancer_Original;
        Ctrain=X(1:342,1:end-1);
        dtrain= X(1:342,end);
        Ctest= X(343:end,1:end-1);
        dtest= X(343:end,end);

     end
      
     if (iData==30)
        load( 'Data_mat\Ionosphere.mat');
        X = Ionosphere;
        Ctrain=X(1:176,1:end-1);
        dtrain= X(1:176,end);
        Ctest= X(177:end,1:end-1);
        dtest= X(177:end,end);

     end
     
     
      
     if (iData==63)
        load('Data_mat\Sonar.mat');
        X = Sonar;
        Ctrain=X(1:104,1:end-1);
        dtrain= X(1:104,end);
        Ctest= X(105:end,1:end-1);
        dtest= X(105:end,end);

     end
     
     if (iData==56)
        load('Data_mat\Promoters.mat');
        X = Promoters;     
        Ctrain=X(1:53,1:end-1);
        dtrain= X(1:53,end);
        Ctest= X(54:end,1:end-1);
        dtest= X(54:end,end);

     end
     
      if (iData==57)
        load('Data_mat\Iris.mat');
        X = Iris;
        Ctrain=X(1:75,1:end-1);
        dtrain= X(1:75,end);
        Ctest= X(76:end,1:end-1);
        dtest= X(76:end,end);

      end
     
      if (iData==51)
        load('Data_mat\Wine.mat');
        X = Wine;
        Ctrain=X(1:89,1:end-1);
        dtrain= X(1:89,end);
        Ctest= X(90:end,1:end-1);
        dtest= X(90:end,end);

      end
      
      if (iData==61)
        load('Data_mat\New_thyroid.mat');
        X = New_thyroid;
        Ctrain=X(1:108,1:end-1);
        dtrain= X(1:108,end);
        Ctest= X(109:end,1:end-1);
        dtest= X(109:end,end);

      end
      
      if (iData==59)
        load('Data_mat_x\breast_cancer.mat');
        X = breast_cancer;
        X = X(randperm(size(X,1)),:);
        Ctrain=X(1:139,1:end-1);
        dtrain= X(1:139,end);
        Ctest= X(140:end,1:end-1);
        dtest= X(140:end,end);

      end
    
    
    for type=1:1
        switch type
           case 1   %%FTBLDM and TBLDM
            %% Train and predict the data

                %%%%%%%-----------------Training the best parameters-----------------%%%%%%%
                TrainRate = 0.5;       % The scale of the tuning set
                t_Train = zeros(N_Times, 1);
                t_Train_ = zeros(N_Times, 1);
                Acc_Predict = zeros(N_Times, 1);
                Acc_Predict_ = zeros(N_Times, 1);
                
                t_Train2 = zeros(N_Times, 1);
                t_Train2_ = zeros(N_Times, 1);
                Acc_Predict2 = zeros(N_Times, 1);
                Acc_Predict2_ = zeros(N_Times, 1);
                
                t_Train3 = zeros(N_Times, 1);
                t_Train3_ = zeros(N_Times, 1);
                Acc_Predict3 = zeros(N_Times, 1);
                Acc_Predict3_ = zeros(N_Times, 1);
                Accuracy = zeros();
                for Times = 1: N_Times

%                     Ctrain0=awgn(Ctrain,0.05); % 高斯噪声
%                     Ctest0=awgn(Ctest,0.05); % 高斯噪声                    
%                     Ctrain0= svdatanorm(Ctrain0,'svpline');
%                     Ctest0= svdatanorm(Ctest0,'scpline');
%                     Samples_Train0 = Ctrain0;
%                     Samples_Predict0 = Ctest0;
%                     Labels_Train = dtrain;                    
%                     Labels_Predict = dtest;                    
                    
                    
                    Ctrain1=awgn(Ctrain,0.05); % 高斯噪声
                    Ctest1=awgn(Ctest,0.05); % 高斯噪声
                    Ctrain1= svdatanorm(Ctrain1,'svpline');
                    Ctest1= svdatanorm(Ctest1,'scpline');
%                     Ctrain = svdatanorm(Ctrain,'svpline');
%                     plot(Ctrain1(dtrain(:,1)==1,1),Ctrain1(dtrain(:,1)==1,2),"o",Ctrain(dtrain(:,1)==1,1),Ctrain(dtrain(:,1)==1,2),"*");
                   
                    Samples_Train = Ctrain1;
                    Samples_Predict = Ctest1;
                    Labels_Train = dtrain;                    
                    Labels_Predict = dtest;   
                                      
                    
                    Ctrain2=awgn(Ctrain,0.1); % 高斯噪声
                    Ctest2=awgn(Ctest,0.1); % 高斯噪声 
                    Ctrain2= svdatanorm(Ctrain2,'svpline');
                    Ctest2= svdatanorm(Ctest2,'scpline');
                    Samples_Train2 = Ctrain2;
                    Samples_Predict2 = Ctest2;                    
                                     
                   
                    Ctrain3=awgn(Ctrain,0.5); % 高斯噪声
                    Ctest3=awgn(Ctest,0.5); % 高斯噪声 
                    Ctrain3= svdatanorm(Ctrain3,'svpline');
                    Ctest3= svdatanorm(Ctest3,'scpline');
                    Samples_Train3 = Ctrain3;
                    Samples_Predict3 = Ctest3;                    
                    


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
                    
                    
                    Best_Acc2 = 0;
                    Best_lambda1_2 = 0;
                    Best_lambda2_2 = 0;
                    Best_lambda3_2 = 0;
                    Best_lambda4_2 = 0;
                    Best_C1_2 = 0;
                    Best_C2_2 = 0;
                    Best_C3_2 = 0;
                    Best_C4_2 = 0;   

                    Best_Acc2_ = 0;
                    Best_lambda1_2_ = 0;
                    Best_lambda2_2_ = 0;
                    Best_lambda3_2_ = 0;
                    Best_lambda4_2_ = 0;
                    Best_C1_2_ = 0;
                    Best_C2_2_ = 0;
                    Best_C3_2_ = 0;
                    Best_C4_2_ = 0; 
                    
                    
                    Best_Acc3 = 0;
                    Best_lambda1_3 = 0;
                    Best_lambda2_3 = 0;
                    Best_lambda3_3 = 0;
                    Best_lambda4_3 = 0;
                    Best_C1_3 = 0;
                    Best_C2_3 = 0;
                    Best_C3_3 = 0;
                    Best_C4_3 = 0;   

                    Best_Acc3_ = 0;
                    Best_lambda1_3_ = 0;
                    Best_lambda2_3_ = 0;
                    Best_lambda3_3_ = 0;
                    Best_lambda4_3_ = 0;
                    Best_C1_3_ = 0;
                    Best_C2_3_ = 0;
                    Best_C3_3_ = 0;
                    Best_C4_3_ = 0; 


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
                                                    
                                                    Samples_A2 = Samples_Train2(I_A,:);
                                                    Samples_B2 = Samples_Train2(I_B,:);
                                                    Samples_A3 = Samples_Train3(I_A,:);
                                                    Samples_B3 = Samples_Train3(I_B,:);

                                                    
                                                    C_s.C1 = C1;
                                                    C_s.C2 = C2;
                                                    S = IFuzzy_MemberShip(Samples_Train, Labels_Train, Kernel, Best_u); 
                                                    C_s.s2 = S.s2;
                                                    C_s.C3 = C3;
                                                    C_s.C4 = C4;
                                                    C_s.s1 = S.s1;     
                                                    
                                                    C_s2.C1 = C1;
                                                    C_s2.C2 = C2;
                                                    S2 = IFuzzy_MemberShip(Samples_Train2, Labels_Train, Kernel, Best_u);
                                                    C_s2.s2 = S2.s2;
                                                    C_s2.C3 = C3;
                                                    C_s2.C4 = C4;
                                                    C_s2.s1 = S2.s1;
                                                    
                                                    C_s3.C1 = C1;
                                                    C_s3.C2 = C2;
                                                    S3 = IFuzzy_MemberShip(Samples_Train3, Labels_Train, Kernel, Best_u);
                                                    C_s3.s2 = S3.s2;
                                                    C_s3.C3 = C3;
                                                    C_s3.C4 = C4;
                                                    C_s3.s1 = S3.s1;


                                                    Outs_Train = Train_FTBLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, lambda1,lambda2 , C_s, Kernel, QPPs_Solver);
                                                    Outs_Train_ = Train_TBLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, lambda1,lambda2 , C_s, Kernel, QPPs_Solver);
                                                    Acc = Predict_FTBLDM(Outs_Train, Samples_Predict,Labels_Predict, Samples_Train);   
                                                    Acc_ = Predict_TBLDM(Outs_Train_, Samples_Predict,Labels_Predict, Samples_Train);
                                                    
                                                    Outs_Train2 = Train_FTBLDM(Samples_A2, Labels_A, Samples_B2,Labels_B, Samples_Train2, lambda1,lambda2 , C_s2, Kernel, QPPs_Solver);
                                                    Outs_Train2_ = Train_TBLDM(Samples_A2, Labels_A, Samples_B2,Labels_B, Samples_Train2, lambda1,lambda2 , C_s2, Kernel, QPPs_Solver);
                                                    Acc2 = Predict_FTBLDM(Outs_Train2, Samples_Predict2,Labels_Predict, Samples_Train2);   
                                                    Acc2_ = Predict_TBLDM(Outs_Train2_, Samples_Predict2,Labels_Predict, Samples_Train2);
                                                    
                                                    Outs_Train3 = Train_FTBLDM(Samples_A3, Labels_A, Samples_B3,Labels_B, Samples_Train3, lambda1,lambda2 , C_s3, Kernel, QPPs_Solver);
                                                    Outs_Train3_ = Train_TBLDM(Samples_A3, Labels_A, Samples_B3,Labels_B, Samples_Train3, lambda1,lambda2 , C_s3, Kernel, QPPs_Solver);
                                                    Acc3 = Predict_FTBLDM(Outs_Train3, Samples_Predict3,Labels_Predict, Samples_Train3);   
                                                    Acc3_ = Predict_TBLDM(Outs_Train3_, Samples_Predict3,Labels_Predict, Samples_Train3);




                                                    Stop_Num = Stop_Num - 1;

                                                    disp([num2str(Stop_Num), ' step(s) remaining.'])


                                                    if Acc>Best_Acc
                                                        Best_Acc = Acc;
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
                                                    if Acc_>Best_Acc_
                                                        Best_Acc_ = Acc_;
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
                                                    
                                                    if Acc2>Best_Acc2
                                                        Best_Acc2 = Acc2;
                                                        Best_lambda1_2 = lambda1;
                                                        Best_lambda2_2 = lambda2;
                                                        Best_lambda3_2 = lambda3;
                                                        Best_lambda4_2 = lambda4;
                                                        Best_C1_2 = C1;
                                                        Best_C2_2 = C2;
                                                        Best_C3_2 = C3;
                                                        Best_C4_2 = C4;                       
                                                        Best_Kernel_2 = Kernel;
                                                    end
                                                    if Acc2_>Best_Acc2_
                                                        Best_Acc2_ = Acc2_;
                                                        Best_lambda1_2_ = lambda1;
                                                        Best_lambda2_2_ = lambda2;
                                                        Best_lambda3_2_ = lambda3;
                                                        Best_lambda4_2_ = lambda4;
                                                        Best_C1_2_ = C1;
                                                        Best_C2_2_ = C2;
                                                        Best_C3_2_ = C3;
                                                        Best_C4_2_ = C4;                       
                                                        Best_Kernel_2_ = Kernel;
                                                    end
                                                    
                                                    if Acc3>Best_Acc3
                                                        Best_Acc3 = Acc3;
                                                        Best_lambda1_3 = lambda1;
                                                        Best_lambda2_3 = lambda2;
                                                        Best_lambda3_3 = lambda3;
                                                        Best_lambda4_3 = lambda4;
                                                        Best_C1_3 = C1;
                                                        Best_C2_3 = C2;
                                                        Best_C3_3 = C3;
                                                        Best_C4_3 = C4;                       
                                                        Best_Kernel_3 = Kernel;
                                                    end
                                                    if Acc3_>Best_Acc3_
                                                        Best_Acc3_ = Acc3_;
                                                        Best_lambda1_3_ = lambda1;
                                                        Best_lambda2_3_ = lambda2;
                                                        Best_lambda3_3_ = lambda3;
                                                        Best_lambda4_3_ = lambda4;
                                                        Best_C1_3_ = C1;
                                                        Best_C2_3_ = C2;
                                                        Best_C3_3_ = C3;
                                                        Best_C4_3_ = C4;                       
                                                        Best_Kernel_3_ = Kernel;
                                                    end

                                                end  % gamma
                                  end

                              end    % lambda2
                        end
                    end    % lambda1


                    BestC_s.C1 = Best_C1;
                    BestC_s.C2 = Best_C2;
                    BestC_s.C3 = Best_C3;
                    BestC_s.C4 = Best_C4;                   
                    S = IFuzzy_MemberShip(Samples_Train, Labels_Train, Kernel, Best_u); 
                    BestC_s.s1 = S.s1;
                    BestC_s.s2 = S.s2;
                    
                    BestC_s_.C1 = Best_C1_;
                    BestC_s_.C2 = Best_C2_;
                    BestC_s_.C3 = Best_C3_;
                    BestC_s_.C4 = Best_C4_;
                    
                   
                    BestC_s2.C1 = Best_C1_2;
                    BestC_s2.C2 = Best_C2_2;
                    BestC_s2.C3 = Best_C3_2;
                    BestC_s2.C4 = Best_C4_2;
                    S2 = IFuzzy_MemberShip(Samples_Train2, Labels_Train, Kernel, Best_u); 
                    BestC_s2.s1 = S2.s1;
                    BestC_s2.s2 = S2.s2;

                    BestC_s2_.C1 = Best_C1_2_;
                    BestC_s2_.C2 = Best_C2_2_;
                    BestC_s2_.C3 = Best_C3_2_;
                    BestC_s2_.C4 = Best_C4_2_;
                    
                    BestC_s3.C1 = Best_C1_3;
                    BestC_s3.C2 = Best_C2_3;
                    BestC_s3.C3 = Best_C3_3;
                    BestC_s3.C4 = Best_C4_3;
                    S3 = IFuzzy_MemberShip(Samples_Train3, Labels_Train, Kernel, Best_u); 
                    BestC_s3.s1 = S3.s1;
                    BestC_s3.s2 = S3.s2;
                    
                    BestC_s3_.C1 = Best_C1_3_;
                    BestC_s3_.C2 = Best_C2_3_;
                    BestC_s3_.C3 = Best_C3_3_;
                    BestC_s3_.C4 = Best_C4_3_;

                    tic         
                    Outs_Train = Train_FTBLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, Best_lambda1, Best_lambda2, BestC_s, Best_Kernel, QPPs_Solver);
                    t_Train(Times) = toc;
                    tic         
                    Outs_Train_ = Train_TBLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, Best_lambda1_, Best_lambda2_, BestC_s_, Best_Kernel_, QPPs_Solver);
                    t_Train_(Times) = toc;
                    
                    tic         
                    Outs_Train2 = Train_FTBLDM(Samples_A2, Labels_A, Samples_B2,Labels_B, Samples_Train2, Best_lambda1_2, Best_lambda2_2, BestC_s2, Best_Kernel_2, QPPs_Solver);
                    t_Train2(Times) = toc;
                    tic         
                    Outs_Train2_ = Train_TBLDM(Samples_A2, Labels_A, Samples_B2,Labels_B, Samples_Train2, Best_lambda1_2_, Best_lambda2_2_, BestC_s2_, Best_Kernel_2_, QPPs_Solver);
                    t_Train2_(Times) = toc;
                    
                    tic         
                    Outs_Train3 = Train_FTBLDM(Samples_A3, Labels_A, Samples_B3,Labels_B, Samples_Train3, Best_lambda1_3, Best_lambda2_3, BestC_s3, Best_Kernel_3, QPPs_Solver);
                    t_Train3(Times) = toc;
                    tic         
                    Outs_Train3_ = Train_TBLDM(Samples_A3, Labels_A, Samples_B3,Labels_B, Samples_Train3, Best_lambda1_3_, Best_lambda2_3_, BestC_s3_, Best_Kernel_3_, QPPs_Solver);
                    t_Train3_(Times) = toc;

                   % Predict the data
                    Acc_Predict(Times) = Best_Acc;
                    Acc_Predict_(Times) = Best_Acc_;
                    
                    Acc_Predict2(Times) = Best_Acc2;
                    Acc_Predict2_(Times) = Best_Acc2_;
                    
                    Acc_Predict3(Times) = Best_Acc3;
                    Acc_Predict3_(Times) = Best_Acc3_;

                end


                %%%%%%%---------------------Save the statiatics---------------------%%%%%%%
%                 Name = Str_Name(10:end-4);
                %                 Loc_Nam = [Location, '\', Name, '.txt'];
                %                 f = fopen(Loc_Nam, 'wt');
%                 disp('FTBLDM');
%                 fprintf('%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train)), '.']);
%                 fprintf('%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(100*Acc_Predict)) '%.']);
% %                 fprintf('%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(100*Acc_Predict)), '.']);
% %                 fprintf('%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
% %                 fprintf('%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);
% 
%                 fprintf('%s\n', 'The Best_gamma is: ',num2str(Best_Kernel.gamma));
%                 fprintf('%s\n', 'The Best_lambda1 is: ',num2str(Best_lambda1));
%                 fprintf('%s\n','The Best_lambda2 is:',num2str(Best_lambda2));
%                 fprintf('%s\n', 'The Best_lambda3 is: ',num2str(Best_lambda3));
%                 fprintf('%s\n','The Best_lambda4 is:',num2str(Best_lambda4));
% 
%                 fprintf('%s\n', 'The Best_C1 is: ',num2str(Best_C1));
%                 fprintf('%s\n', 'The Best_C2 is: ',num2str(Best_C2));
%                 fprintf('%s\n', 'The Best_C3 is: ',num2str(Best_C3));
%                 fprintf('%s\n','The Best_C4 is:',num2str(Best_C4));
% 
%                 fprintf('%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train_)), '.']);
%                 fprintf('%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(100*Acc_Predict_)) '%.']);
% %                 fprintf('%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(100*Acc_Predict)), '.']);
% %                 fprintf('%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
% %                 fprintf('%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);
% 
%                 disp('TBLDM');
%                 fprintf('%s\n', 'The Best_gamma is: ',num2str(Best_Kernel_.gamma));
%                 fprintf('%s\n', 'The Best_lambda1 is: ',num2str(Best_lambda1_));
%                 fprintf('%s\n','The Best_lambda2 is:',num2str(Best_lambda2_));
%                 fprintf('%s\n', 'The Best_lambda3 is: ',num2str(Best_lambda3_));
%                 fprintf('%s\n','The Best_lambda4 is:',num2str(Best_lambda4_));
% 
%                 fprintf('%s\n', 'The Best_C1 is: ',num2str(Best_C1_));
%                 fprintf('%s\n', 'The Best_C2 is: ',num2str(Best_C2_));
%                 fprintf('%s\n', 'The Best_C3 is: ',num2str(Best_C3_));
%                 fprintf('%s\n','The Best_C4 is:',num2str(Best_C4_));                

                temp = [mean(t_Train),mean(100*Acc_Predict),...
                   Best_Kernel.gamma, Best_lambda1,Best_lambda2,Best_lambda3,Best_lambda4,Best_C1,Best_C2,Best_C3,Best_C4];
                xlswrite('result_noise.xlsx',temp,['B',num2str(iData-37),':L',num2str(iData-36)]);

                temp_ = [mean(t_Train_),mean(100*Acc_Predict_),...
                   Best_Kernel_.gamma, Best_lambda1_,Best_lambda2_,Best_lambda3_,Best_lambda4_,Best_C1_,Best_C2_,Best_C3_,Best_C4_];
                xlswrite('result_noise.xlsx',temp_,['N',num2str(iData-37),':X',num2str(iData-36)]); 
                
                temp2 = [mean(t_Train2),mean(100*Acc_Predict2),...
                   Best_Kernel_2.gamma, Best_lambda1_2,Best_lambda2_2,Best_lambda3_2,Best_lambda4_2,Best_C1_2,Best_C2_2,Best_C3_2,Best_C4_2];
                xlswrite('result_noise.xlsx',temp2,['B',num2str(iData-17),':L',num2str(iData-17)]);

                temp2_ = [mean(t_Train2_),mean(100*Acc_Predict2_),...
                   Best_Kernel_2_.gamma, Best_lambda1_2_,Best_lambda2_2_,Best_lambda3_2_,Best_lambda4_2_,Best_C1_2_,Best_C2_2_,Best_C3_2_,Best_C4_2_];
                xlswrite('result_noise.xlsx',temp2_,['N',num2str(iData-18),':X',num2str(iData-17)]);
                
                
                temp3 = [mean(t_Train3),mean(100*Acc_Predict3),...
                   Best_Kernel_3.gamma, Best_lambda1_3,Best_lambda2_3,Best_lambda3_3,Best_lambda4_3,Best_C1_3,Best_C2_3,Best_C3_3,Best_C4_3];
                xlswrite('result_noise.xlsx',temp3,['B',num2str(iData+1),':L',num2str(iData+2)]);

                temp3_ = [mean(t_Train3_),mean(100*Acc_Predict3_),...
                   Best_Kernel_3_.gamma, Best_lambda1_3_,Best_lambda2_3_,Best_lambda3_3_,Best_lambda4_3_,Best_C1_3_,Best_C2_3_,Best_C3_3_,Best_C4_3_];
                xlswrite('result_noise.xlsx',temp3_,['N',num2str(iData+1),':X',num2str(iData+2)]);

           case 2   %%TLDM
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

                    Ctrain=awgn(Ctrain,0.1); % 高斯噪声
                    Ctest=awgn(Ctest,0.1); % 高斯噪声 
                    Ctrain= svdatanorm(Ctrain,'svpline');
                    Ctest= svdatanorm(Ctest,'scpline');
                    Samples_Train = Ctrain;
                    Labels_Train = dtrain;
                    Samples_Predict = Ctest;
                    Labels_Predict = dtest;

                    Best_Acc = 0;
                    Best_lambda1 = 0;
                    Best_lambda2 = 0;
                    Best_lambda3 = 0;
                    Best_lambda4 = 0;
                    Best_C1 = 0;
                    Best_C2 = 0;
                    Best_C3 = 0;
                    Best_C4 = 0;                    


                    for ith_lambda1 = 1:length(lambda1_Interval)    % lambda1
                        lambda1 = lambda1_Interval(ith_lambda1);    % lambda1
                        lambda3 = lambda1;
                        for ith_lambda2 = 1:length(lambda2_Interval)
                          lambda2 = lambda2_Interval(ith_lambda2);
                          lambda4 = lambda2;


                                  for ith_C3 = 1:length(C3_Interval) 
                                       C3 = C3_Interval(ith_C3);    % lambda3
                                       C4 = C3;

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


                                                    C_s.s2 = Fuzzy_MemberShip(Samples_B, Labels_B, Kernel, Best_u);
                                                    C_s.C3 = C3;
                                                    C_s.C4 = C4;
                                                    C_s.s1 = Fuzzy_MemberShip(Samples_A, Labels_A, Kernel, Best_u);                                   


                                                    Outs_Train = Train_TLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, lambda1,lambda2 , C_s, Kernel, QPPs_Solver);

                                                    Acc = Predict_TLDM(Outs_Train, Samples_Predict,Labels_Predict, Samples_Train);   


                                                    Stop_Num = Stop_Num - 1;

                                                    disp([num2str(Stop_Num), ' step(s) remaining.'])


                                                    if Acc>Best_Acc
                                                        Best_Acc = Acc;
                                                        Best_lambda1 = lambda1;
                                                        Best_lambda2 = lambda2;
                                                        Best_lambda3 = lambda3;
                                                        Best_lambda4 = lambda4;
                                                        Best_C3 = C3;
                                                        Best_C4 = C4;                       
                                                        Best_Kernel = Kernel;
                                                    end

                                                end  % gamma
                                  end

                        end
                    end    % lambda1



                    BestC_s.C3 = Best_C3;
                    BestC_s.C4 = Best_C4;
                    BestC_s.s1 = Fuzzy_MemberShip(Samples_A, Labels_A, Best_Kernel, Best_u);
                    BestC_s.s2 = Fuzzy_MemberShip(Samples_B, Labels_B, Best_Kernel, Best_u);


                    tic         
                    Outs_Train = Train_TLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, Best_lambda1, Best_lambda2, BestC_s, Best_Kernel, QPPs_Solver);
                    t_Train(Times) = toc;

                   % Predict the data
                    Acc_Predict(Times) = Best_Acc;

                end


                %%%%%%%---------------------Save the statiatics---------------------%%%%%%%
%                 Name = Str_Name(10:end-4);
                %                 Loc_Nam = [Location, '\', Name, '.txt'];
                %                 f = fopen(Loc_Nam, 'wt');
                disp('TLDM');
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

                fprintf('%s\n', 'The Best_C3 is: ',num2str(Best_C3));
                fprintf('%s\n','The Best_C4 is:',num2str(Best_C4));
            
                temp = [mean(t_Train),mean(100*Acc_Predict),...
                   Best_Kernel.gamma, Best_lambda1,Best_lambda2,Best_lambda3,Best_lambda4,Best_C3,Best_C4];
                xlswrite('result_noise.xlsx',temp,['BB',num2str(iData-37),':BJ',num2str(iData-37)]);
                
            case 3  %%TBSVM
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

                    Ctrain=awgn(Ctrain,0.1); % 高斯噪声
                    Ctest=awgn(Ctest,0.1); % 高斯噪声 
                    Ctrain= svdatanorm(Ctrain,'svpline');
                    Ctest= svdatanorm(Ctest,'scpline');
                    Samples_Train = Ctrain;
                    Labels_Train = dtrain;
                    Samples_Predict = Ctest;
                    Labels_Predict = dtest;

                    Best_Acc = 0;
                    Best_lambda1 = 0;
                    Best_lambda2 = 0;
                    Best_lambda3 = 0;
                    Best_lambda4 = 0;
                    Best_C1 = 0;
                    Best_C2 = 0;
                    Best_C3 = 0;
                    Best_C4 = 0;                    


%                     for ith_lambda1 = 1:length(lambda1_Interval)    % lambda1
%                         lambda1 = lambda1_Interval(ith_lambda1);    % lambda1
%                         lambda3 = lambda1;
%                         for ith_lambda2 = 1:length(lambda2_Interval)
%                           lambda2 = lambda2_Interval(ith_lambda2);
%                           lambda4 = lambda2;
                              for ith_C1 = 1:length(C1_Interval) 
                                  C1 = C1_Interval(ith_C1);    % lambda3
                                  C2 = C1;

                                  for ith_C3 = 1:length(C3_Interval) 
                                       C3 = C3_Interval(ith_C3);    % lambda3
                                       C4 = C3;

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


                                                    Parameter.ker = 'rbf';
                                                    Parameter.CC = C3;
                                                    Parameter.CR = C1;
                                                    Parameter.p1 = Kernel.gamma;
                                                    Parameter.algorithm = 'QP';
                                                    Parameter.showplots = false;                                  


                                                    [tbsvm_Substruct] = tbsvmtrain(Samples_Train,Labels_Train,Parameter);

                                                    [Acc]= tbsvmclass(tbsvm_Substruct,Samples_Predict,Labels_Predict);   


                                                    Stop_Num = Stop_Num - 1;

                                                    disp([num2str(Stop_Num), ' step(s) remaining.'])


                                                    if Acc>Best_Acc
                                                        Best_Acc = Acc;
%                                                         Best_lambda1 = lambda1;
%                                                         Best_lambda2 = lambda2;
%                                                         Best_lambda3 = lambda3;
%                                                         Best_lambda4 = lambda4;
                                                        Best_C1 = C1;
                                                        Best_C2 = C2;
                                                        Best_C3 = C3;
                                                        Best_C4 = C4;                       
                                                        Best_Kernel = Kernel;
                                                    end

                                                end  % gamma
                                  end
                              end
%                         end
%                     end    % lambda1



                    Parameter.ker = 'rbf';
                    Parameter.CC = Best_C3;
                    Parameter.CR = Best_C1;
                    Parameter.algorithm = 'QP';
                    Parameter.p1 = Best_Kernel.gamma;


                    tic         
                    [tbsvm_struct] = tbsvmtrain(Samples_Train,Labels_Train,Parameter);
                    t_Train(Times) = toc;

                   % Predict the data
                    Acc_Predict(Times) = Best_Acc;

                end


                %%%%%%%---------------------Save the statiatics---------------------%%%%%%%
%                 Name = Str_Name(10:end-4);
                %                 Loc_Nam = [Location, '\', Name, '.txt'];
                %                 f = fopen(Loc_Nam, 'wt');
                disp('TBSVM');
                fprintf('%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train)), '.']);
                fprintf('%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(Acc_Predict)) '%.']);
%                 fprintf('%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(100*Acc_Predict)), '.']);
%                 fprintf('%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
%                 fprintf('%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);

                fprintf('%s\n', 'The Best_gamma is: ',num2str(Best_Kernel.gamma));
%                 fprintf('%s\n', 'The Best_lambda1 is: ',num2str(Best_lambda1));
%                 fprintf('%s\n','The Best_lambda2 is:',num2str(Best_lambda2));
%                 fprintf('%s\n', 'The Best_lambda3 is: ',num2str(Best_lambda3));
%                 fprintf('%s\n','The Best_lambda4 is:',num2str(Best_lambda4));
                fprintf('%s\n', 'The Best_C1 is: ',num2str(Best_C1));
                fprintf('%s\n','The Best_C2 is:',num2str(Best_C2));
                fprintf('%s\n', 'The Best_C3 is: ',num2str(Best_C3));
                fprintf('%s\n','The Best_C4 is:',num2str(Best_C4));
            
                temp = [mean(t_Train),mean(Acc_Predict),...
                   Best_Kernel.gamma,Best_C1,Best_C2,Best_C3,Best_C4];
                xlswrite('result_noise.xlsx',temp,['BL',num2str(iData-37),':BR',num2str(iData-37)]);        
            case 4   %%FTSVM and TSVM
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

                    Ctrain=awgn(Ctrain,0.1); % 高斯噪声
                    Ctest=awgn(Ctest,0.1); % 高斯噪声 
                    Ctrain= svdatanorm(Ctrain,'svpline');
                    Ctest= svdatanorm(Ctest,'scpline');
                    Samples_Train = Ctrain;
                    Labels_Train = dtrain;
                    Samples_Predict = Ctest;
                    Labels_Predict = dtest;

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


%                     for ith_lambda1 = 1:length(lambda1_Interval)    % lambda1
%                         lambda1 = lambda1_Interval(ith_lambda1);    % lambda1
%                         lambda3 = lambda1;
%                         for ith_lambda2 = 1:length(lambda2_Interval)
%                           lambda2 = lambda2_Interval(ith_lambda2);
%                           lambda4 = lambda2;
% 
% 
%                                for ith_C1 = 1:length(C1_Interval)    % lambda3
%                                    C1 = C1_Interval(ith_C1);    % lambda3
%                                    C2 = C1;
                                  for ith_C1 = 1:length(C1_Interval) 
                                       C1 = C1_Interval(ith_C1);    % lambda3
                                       C2 = C1;

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
                                                    s = Fuzzy_MemberShip(Samples_Train, Labels_Train, Kernel, Best_u);
                                                    C_s.s2 = s.s2;
%                                                     C_s.C3 = C3;
%                                                     C_s.C4 = C4;
                                                    C_s.s1 = s.s2;                                   


                                                    Outs_Train = Train_FTSVM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train,C_s, Kernel, QPPs_Solver);
                                                    Outs_Train_ = Train_TSVM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train,C_s, Kernel, QPPs_Solver);


                                                    Acc = Predict_FTSVM(Outs_Train, Samples_Predict,Labels_Predict, Samples_Train);   
                                                    Acc_ = Predict_TSVM(Outs_Train_, Samples_Predict,Labels_Predict, Samples_Train);


                                                    Stop_Num = Stop_Num - 1;

                                                    disp([num2str(Stop_Num), ' step(s) remaining.'])


                                                    if Acc>Best_Acc
                                                        Best_Acc = Acc;
%                                                         Best_lambda1 = lambda1;
%                                                         Best_lambda2 = lambda2;
%                                                         Best_lambda3 = lambda3;
%                                                         Best_lambda4 = lambda4;
                                                        Best_C1 = C1;
                                                        Best_C2 = C2;
%                                                         Best_C3 = C3;
%                                                         Best_C4 = C4;                       
                                                        Best_Kernel = Kernel;
                                                    end

                                                    if Acc_>Best_Acc_
                                                        Best_Acc_ = Acc_;
%                                                         Best_lambda1_ = lambda1;
%                                                         Best_lambda2_ = lambda2;
%                                                         Best_lambda3_ = lambda3;
%                                                         Best_lambda4_ = lambda4;
                                                        Best_C1_ = C1;
                                                        Best_C2_ = C2;
%                                                         Best_C3_ = C3;
%                                                         Best_C4_ = C4;                       
                                                        Best_Kernel_ = Kernel;
                                                    end

                                                end  % gamma
                                  end

%                               end    % lambda2
%                         end
%                     end    % lambda1


                    BestC_s.C1 = Best_C1;
                    BestC_s.C2 = Best_C2;
%                     BestC_s.C3 = Best_C3;
%                     BestC_s.C4 = Best_C4;
                    Bests = Fuzzy_MemberShip(Samples_Train, Labels_Train, Kernel, Best_u); 
                    BestC_s.s1 = Bests.s1;
                    BestC_s.s2 = Bests.s2;

                    BestC_s_.C1 = Best_C1_;
                    BestC_s_.C2 = Best_C2_;
                    BestC_s_.C3 = Best_C3_;
                    BestC_s_.C4 = Best_C4_;

                    tic         
                    Outs_Train = Train_FTSVM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, BestC_s, Best_Kernel, QPPs_Solver);
                    t_Train(Times) = toc;

                    tic         
                    Outs_Train_ = Train_TSVM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, BestC_s_, Best_Kernel_, QPPs_Solver);
                    t_Train_(Times) = toc;

                   % Predict the data
                    Acc_Predict(Times) = Best_Acc;
                    Acc_Predict_(Times) = Best_Acc_;

                end


                %%%%%%%---------------------Save the statiatics---------------------%%%%%%%
%                 Name = Str_Name(10:end-4);
                %                 Loc_Nam = [Location, '\', Name, '.txt'];
                %                 f = fopen(Loc_Nam, 'wt');
                disp('FTBLDM');
                fprintf('%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train)), '.']);
                fprintf('%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(100*Acc_Predict)) '%.']);
%                 fprintf('%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(100*Acc_Predict)), '.']);
%                 fprintf('%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
%                 fprintf('%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);

                fprintf('%s\n', 'The Best_gamma is: ',num2str(Best_Kernel.gamma));
%                 fprintf('%s\n', 'The Best_lambda1 is: ',num2str(Best_lambda1));
%                 fprintf('%s\n','The Best_lambda2 is:',num2str(Best_lambda2));
%                 fprintf('%s\n', 'The Best_lambda3 is: ',num2str(Best_lambda3));
%                 fprintf('%s\n','The Best_lambda4 is:',num2str(Best_lambda4));

                fprintf('%s\n', 'The Best_C1 is: ',num2str(Best_C1));
                fprintf('%s\n', 'The Best_C2 is: ',num2str(Best_C2));
%                 fprintf('%s\n', 'The Best_C3 is: ',num2str(Best_C3));
%                 fprintf('%s\n','The Best_C4 is:',num2str(Best_C4));

                fprintf('%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train_)), '.']);
                fprintf('%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(100*Acc_Predict_)) '%.']);
%                 fprintf('%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(100*Acc_Predict)), '.']);
%                 fprintf('%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
%                 fprintf('%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);

                disp('TBLDM');
                fprintf('%s\n', 'The Best_gamma is: ',num2str(Best_Kernel_.gamma));
%                 fprintf('%s\n', 'The Best_lambda1 is: ',num2str(Best_lambda1_));
%                 fprintf('%s\n','The Best_lambda2 is:',num2str(Best_lambda2_));
%                 fprintf('%s\n', 'The Best_lambda3 is: ',num2str(Best_lambda3_));
%                 fprintf('%s\n','The Best_lambda4 is:',num2str(Best_lambda4_));

                fprintf('%s\n', 'The Best_C1 is: ',num2str(Best_C1_));
                fprintf('%s\n', 'The Best_C2 is: ',num2str(Best_C2_));
%                 fprintf('%s\n', 'The Best_C3 is: ',num2str(Best_C3_));
%                 fprintf('%s\n','The Best_C4 is:',num2str(Best_C4_));                

                temp = [mean(t_Train),mean(100*Acc_Predict),...
                   Best_Kernel.gamma,Best_C1,Best_C2];
                xlswrite('result_noise.xlsx',temp,['AP',num2str(iData-37),':AT',num2str(iData-37)]);

                temp_ = [mean(t_Train_),mean(100*Acc_Predict_),...
                   Best_Kernel_.gamma,Best_C1_,Best_C2_,];
                xlswrite('result_noise.xlsx',temp_,['AV',num2str(iData-37),':AZ',num2str(iData-37)]);    
        end
    end
end   
