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
gamma_Interval = 2.^(-5:2);
lambda1_Interval = 2.^(-8:4);
lambda2_Interval = 2.^(-8:4);
C1_Interval = 2.^(-8:4);
C3_Interval = 2.^(-8:4);


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
for iData =99:99
%                 Str_Name = Str(iData).Name;
%                 Output = load(Str_Name);
%                 Data_Name = fieldnames(Output);   % A struct data
%                 Data_Original = getfield(Output, Data_Name{1}); % Abstract the data
%                 % Normalization
%                 Data_Original = [mapminmax(Data_Original(:, 1:end-1)', 0, 1)', Data_Original(:, end)]; % Map the original data to value between [0, 1] by colum
% %                 M_Original = size(Data_Original, 1);
% %                 Data_Original = Data_Original(randperm(M_Original), :);

    if (iData==1)
        load('Data_mat_n\monks_1_train.txt');
        load('Data_mat_n\monks_1_test.txt');
        Ctrain= monks_1_train(:,2:end);
        dtrain=  monks_1_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= monks_1_test(:,2:end);
        dtest=  monks_1_test(:,1);
        dtest(find(dtest==0))=-1;
    end

    if(iData==2)
        load('Data_mat_n\monks_2_train.txt');
        load('Data_mat_n\monks_2_test.txt');
        Ctrain= monks_2_train(:,2:end);
        dtrain=  monks_2_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= monks_2_test(:,2:end);
        dtest=  monks_2_test(:,1);
        dtest(find(dtest==0))=-1;
    end

    if (iData==3)
        load('Data_mat_n\monks_3_train.txt');
        load('Data_mat_n\monks_3_test.txt');
        Ctrain= monks_3_train(:,2:end);
        dtrain=  monks_3_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= monks_3_test(:,2:end);
        dtest=  monks_3_test(:,1);
        dtest(find(dtest==0))=-1;
    end
    
     if (iData==4)
        load( 'Data_mat_x\tic_tac_toe.mat');
        X = tic_tac_toe;
        Ctrain=X(1:479,1:end-1);
        dtrain= X(1:479,end);
        Ctest= X(480:end,1:end-1);
        dtest= X(480:end,end);

     end
     if (iData==5)
        load('Data_mat\Promoters.mat');
        X = Promoters;     
        Ctrain=X(1:53,1:end-1);
        dtrain= X(1:53,end);
        Ctest= X(54:end,1:end-1);
        dtest= X(54:end,end);

     end
     
     if (iData==6)
        load('Data_mat\Iris.mat');
        X = Iris;
        Ctrain=X(1:75,1:end-1);
        dtrain= X(1:75,end);
        Ctest= X(76:end,1:end-1);
        dtest= X(76:end,end);

     end
      
     if (iData==7)
        load('Data_mat\Wine.mat');
        X = Wine;
        Ctrain=X(1:89,1:end-1);
        dtrain= X(1:89,end);
        Ctest= X(90:end,1:end-1);
        dtest= X(90:end,end);

     end
      
      if (iData==8)
        load( 'Data_mat\Haberman.mat');
        X = Haberman;
        Ctrain=X(1:153,1:end-1);
        dtrain= X(1:153,end);
        Ctest= X(154:end,1:end-1);
        dtest= X(154:end,end);
        
      end
     
      if (iData==9)
        load('Data_mat_x\breast_cancer.mat');
        X = breast_cancer;
        X = X(randperm(size(X,1)),:);
        Ctrain=X(1:139,1:end-1);
        dtrain= X(1:139,end);
        Ctest= X(140:end,1:end-1);
        dtest= X(140:end,end);

      end
      
       if (iData==10)
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

    if (iData==11)
        load('Data_mat_n\Wpbc.mat');
        X = Wpbc;
        Ctrain=X(1:90,1:end-1);
        dtrain= X(1:90,end);
        Ctest= X(91:end,1:end-1);
        dtest= X(91:end,end);

    end
    
    if (iData==12)
        load( 'Data_mat\Breast_cancer_Original.mat');
        X = Breast_cancer_Original;
        Ctrain=X(1:342,1:end-1);
        dtrain= X(1:342,end);
        Ctest= X(343:end,1:end-1);
        dtest= X(343:end,end);

    end
    
     if (iData==13)
        load('Data_mat\New_thyroid.mat');
        X = New_thyroid;
        Ctrain=X(1:108,1:end-1);
        dtrain= X(1:108,end);
        Ctest= X(109:end,1:end-1);
        dtest= X(109:end,end);

     end
      
      if (iData==14)
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
    
      if (iData==15)
        load('Data_mat_x\heart_statlog.mat');
        X = heart_statlog;
        Ctrain=X(1:135,1:end-1);
        dtrain= X(1:135,end);
        Ctest= X(136:end,1:end-1);
        dtest= X(136:end,end);

     end
     
     if (iData==16)
        load( 'Data_mat\Hepatitis.mat');
        X = Hepatitis;
        Ctrain=X(1:78,1:end-1);
        dtrain= X(1:78,end);
        Ctest= X(79:end,1:end-1);
        dtest= X(79:end,end);

     end
     
     if (iData==17)
        load('Data_mat\Sonar.mat');
        X = Sonar;
        Ctrain=X(1:104,1:end-1);
        dtrain= X(1:104,end);
        Ctest= X(105:end,1:end-1);
        dtest= X(105:end,end);

     end
    
     if (iData==18)
        load( 'Data_mat\Glass.mat');
        X = Glass;
        Ctrain=X(1:107,1:end-1);
        dtrain= X(1:107,end);
        Ctest= X(108:end,1:end-1);
        dtest= X(108:end,end);

     end

    if(iData==19)
        load('Data_mat_n\plrx.txt');
        X=plrx(:,1:end-1);
        Y= plrx(:,end);
        Y(find(Y==2))=-1;
        Ctrain=X(1:100,:);
        dtrain= Y(1:100,:);
        Ctest= X(101:end,:);
        dtest= Y(101:end,:);

    end
    
     if (iData==20)
        load('Data_mat_x\credit_a.mat');
        X = credit_a;        
        Ctrain=X(1:327,1:end-1);
        dtrain= X(1:327,end);
        Ctest= X(328:end,1:end-1);
        dtest= X(328:end,end);

     end
     
      if (iData==21)
        load( 'Data_mat\Ionosphere.mat');
        X = Ionosphere;
        Ctrain=X(1:176,1:end-1);
        dtrain= X(1:176,end);
        Ctest= X(177:end,1:end-1);
        dtest= X(177:end,end);

     end
    
     if (iData==22)
        load('Data_mat\Australian.mat');
        X = Australian;
        Ctrain=X(1:395,1:end-1);
        dtrain= X(1:395,end);
        Ctest= X(396:end,1:end-1);
        dtest= X(396:end,end);

     end
      

    if(iData==23)
        data=xlsread('Data_mat_n\fertility_Diagnosis.xlsx');
        X= data(:,1:end-1);
        Y= data(:,end);
        Y(Y==0)=-1;
        Ctrain=X(1:50,:);
        dtrain= Y(1:50,:);
        Ctest= X(51:end,:);
        dtest= Y(51:end,:);
    end
    
    

    if (iData==24)
        load('Data_mat_n\SPECT_train.txt')
        load('Data_mat_n\SPECT_test.txt');
        Ctrain = SPECT_train(:,2:end);
        dtrain=  SPECT_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= SPECT_test(:,2:end);
        dtest=  SPECT_test(:,1);
        dtest(find(dtest==0))=-1;
    end  
     

     
    if(iData==25)
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
    
    if (iData==26)
        load('Data_mat_n\votes.mat')
        X=votes(:,2:end);
        Y= votes(:,1);
        Y(find(Y==2))=-1;
        Ctrain=X(1:200,:);
        dtrain= Y(1:200,:);
        Ctest= X(201:end,:);
        dtest= Y(201:end,:);
     end
    
    
    

   
     
     
     
%      if (iData==24)
%         load('Data_mat\Ripley_Train.mat')
%         load('Data_mat\Ripley_Predict.mat');
%         Ctrain = Ripley_Train(:,1:end-1);
%         dtrain=  Ripley_Train(:,end);
%         Ctest= Ripley_Predict(:,1:end-1);
%         dtest=  Ripley_Predict(:,end);  
%         
%     end

    
      
%      if (iData==26)
%         load( 'Data_mat\CMC.mat');
%         X = CMC;
%         Ctrain=X(1:737,1:end-1);
%         dtrain= X(1:737,end);
%         Ctest= X(738:end,1:end-1);
%         dtest= X(738:end,end);
% 
%      end
      
    
      
    
    
      
     
     
     
     
      
     
      
      
      
      
      
      
      
      
      
      
     
     
     
      
     
      
      if (iData==99) %artificial

        data_train=[data1(1:150,:);data2(1:150,:)];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(151:end,:);data2(151:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
      end  
      
      if (iData==100) %artificial
        N_data_noise = data_noise(1:10,:);

        data_train=[data1(1:100,:);data2(1:100,:);data_noise];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(101:end,:);data2(101:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
      end
      
      if (iData==101) %artificial
        N_data_noise = data_noise(1:20,:);

        data_train=[data1(1:100,:);data2(1:100,:);data_noise];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(101:end,:);data2(101:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
      end
      
      
      if (iData==102) %artificial
        N_data_noise = data_noise(1:30,:);

        data_train=[data1(1:100,:);data2(1:100,:);data_noise];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(101:end,:);data2(101:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
      end
      
      if (iData==103) %artificial
        N_data_noise = data_noise(1:40,:);

        data_train=[data1(1:100,:);data2(1:100,:);data_noise];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(101:end,:);data2(101:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
      end
      
      if (iData==104) %artificial
        N_data_noise = data_noise(1:50,:);

        data_train=[data1(1:100,:);data2(1:100,:);data_noise];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(101:end,:);data2(101:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
      end
      
      if (iData==105) %artificial
        N_data_noise = data_noise(1:60,:);

        data_train=[data1(1:100,:);data2(1:100,:);data_noise];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(101:end,:);data2(101:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
      end
      
      if (iData==106) %artificial
        N_data_noise = data_noise(1:70,:);

        data_train=[data1(1:100,:);data2(1:100,:);data_noise];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(101:end,:);data2(101:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
      end
      
      if (iData==107) %artificial
        N_data_noise = data_noise(1:80,:);

        data_train=[data1(1:100,:);data2(1:100,:);data_noise];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(101:end,:);data2(101:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
      end
      
      if (iData==108) %artificial
        N_data_noise = data_noise(1:90,:);

        data_train=[data1(1:100,:);data2(1:100,:);data_noise];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(101:end,:);data2(101:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
      end
      
      if (iData==109) %artificial
        N_data_noise = data_noise(1:100,:);

        data_train=[data1(1:100,:);data2(1:100,:);data_noise];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(101:end,:);data2(101:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
      end
      
      if (iData==110) %artificial
        N_data_noise = data_noise(1:110,:);

        data_train=[data1(1:100,:);data2(1:100,:);data_noise];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(101:end,:);data2(101:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
      end
      
      if (iData==111) %artificial
        N_data_noise = data_noise(1:120,:);

        data_train=[data1(1:100,:);data2(1:100,:);data_noise];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(101:end,:);data2(101:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
      end
      
      if (iData==112) %artificial
        N_data_noise = data_noise(1:130,:);

        data_train=[data1(1:100,:);data2(1:100,:);data_noise];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(101:end,:);data2(101:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
      end
      
      if (iData==113) %artificial
        
        N_data_noise = data_noise(1:140,:);

        data_train=[data1(1:100,:);data2(1:100,:);data_noise];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(101:end,:);data2(101:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
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
                
                Accuracy = zeros();
                for Times = 1: N_Times

                    Ctrain=awgn(Ctrain,0.1); % 高斯噪声
                    Ctest=awgn(Ctest,0.1); % 高斯噪声  
                    
                    Ctrain= svdatanorm(Ctrain,'svpline');
                    Ctest= svdatanorm(Ctest,'scpline');
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

                    
                    Best_lambda1_ = 0;
                    Best_lambda2_ = 0;
                    Best_lambda3_ = 0;
                    Best_lambda4_ = 0;
                    Best_C1_ = 0;
                    Best_C2_ = 0;
                    Best_C3_ = 0;
                    Best_C4_ = 0;  
                    
                    
                    C3_ACC = ones(1,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                    C3_lambda1 = ones(length(lambda1_Interval),1);
                    C3_lambda2 = ones(length(lambda2_Interval),1);
                    C3_C1 = ones(length(C1_Interval),1);
                    C3_C3 = ones(length(C3_Interval),1);
                    Gam = ones(length(gamma_Interval),1);
                    
                    C3_ACC_ = ones(1,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                    C3_lambda1_ = ones(length(lambda1_Interval),1);
                    C3_lambda2_ = ones(length(lambda2_Interval),1);
                    C3_C1_ = ones(length(C1_Interval),1);
                    C3_C3_ = ones(length(C3_Interval),1);
                    Gam_ = ones(length(gamma_Interval),1);
                    
                                       
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



%                                                             membership(Samples_Train', Labels_Train')
                                                            s = IFuzzy_MemberShip(Samples_Train, Labels_Train, Kernel, Best_u); 
                                                            C_s.C1 = C1;
                                                            C_s.C2 = C2;
                                                            C_s.s1 = s.s1;
                                                            C_s.s2 = s.s2;
                                                            C_s.C3 = C3;
                                                            C_s.C4 = C4;



                                                            Outs_Train = Train_FTBLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, lambda1,lambda2 , C_s, Kernel, QPPs_Solver);
                                                            Outs_Train_ = Train_TBLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, lambda1,lambda2 , C_s, Kernel, QPPs_Solver);
                                                            Acc = Predict_FTBLDM(Outs_Train, Samples_Predict,Labels_Predict, Samples_Train);   
                                                            Acc_ = Predict_TBLDM(Outs_Train_, Samples_Predict,Labels_Predict, Samples_Train);


                                                           Stop_Num = Stop_Num - 1;

                                                            disp([num2str(Stop_Num), ' step(s) remaining.'])


                                                            if Acc>Best_Acc
                                                                Best_Acc = Acc;
                                                                Best_lambda1 = lambda1;
                                                                Best_lambda2 = lambda2; 
                                                                Best_C1 = C1;
                                                                Best_C3 = C3;                      
                                                                Best_Kernel = Kernel;
                                                            end
                                                            if Acc_>Best_Acc_
                                                                Best_Acc_ = Acc_;
                                                                Best_lambda1_ = lambda1;
                                                                Best_lambda2_ = lambda2;
                                                                Best_C1_ = C1;
                                                                Best_C3_ = C3;                      
                                                                Best_Kernel_ = Kernel;
                                                            end


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

                            C3_ACC_(ith_C3) = Best_Acc_;
                            C3_lambda1_(ith_C3) = Best_lambda1_;
                            C3_lambda2_(ith_C3) = Best_lambda2_;
                            C3_C1_(ith_C3) = Best_C1_;
                            C3_C3_(ith_C3) = Best_C3_;
                            Gam_(ith_C3) = Best_Kernel_.gamma;
                            

                            
                            
                    end

                    [Best_Acc,i]=max(C3_ACC);
                    [Best_Acc_,j]=max(C3_ACC_);
                    
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

                    BestC_s_.C1 = C3_C1_(j);
                    BestC_s_.C2 = C3_C1_(j);
                    BestC_s_.C3 = C3_C3_(j);
                    BestC_s_.C4 = C3_C3_(j);
                    Best_lambda1_ = C3_lambda1_(j);
                    Best_lambda2_ = C3_lambda2_(j);
                    Best_Kernel_.gamma = Gam_(j);
                    
                    
                    tic         
                    Outs_Train = Train_FTBLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, Best_lambda1, Best_lambda2, BestC_s, Best_Kernel, QPPs_Solver);
                    t_Train(Times) = toc;
                    tic         
                    Outs_Train_ = Train_TBLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, Best_lambda1_, Best_lambda2_, BestC_s_, Best_Kernel_, QPPs_Solver);
                    t_Train_(Times) = toc;
                    

                   % Predict the data
                    Acc_Predict(Times) = Best_Acc;
                    Acc_Predict_(Times) = Best_Acc_;
                    


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
                   Best_Kernel.gamma, Best_lambda1,Best_lambda2,BestC_s.C1,BestC_s.C3];
                xlswrite('result.xlsx',temp,['C',num2str(iData+2),':I',num2str(iData+2)]);
                xlswrite('result.xlsx',C3_ACC,['C',num2str(iData+30),':O',num2str(iData+30)]);
                

                temp_ = [mean(t_Train_),mean(100*Acc_Predict_),...
                   Best_Kernel_.gamma, Best_lambda1_,Best_lambda2_,BestC_s_.C1,BestC_s_.C3];
                xlswrite('result.xlsx',temp_,['K',num2str(iData+2),':Q',num2str(iData+2)]); 
                xlswrite('result.xlsx',C3_ACC_,['C',num2str(iData+52),':O',num2str(iData+52)]);

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

                    Best_lambda1 = 0;
                    Best_lambda2 = 0;
                    Best_lambda3 = 0;
                    Best_lambda4 = 0;
                    Best_C3 = 0;
                    Best_C4 = 0;  
                    
                    C3_ACC = ones(1,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                    C3_lambda1 = ones(length(lambda1_Interval),1);
                    C3_lambda2 = ones(length(lambda2_Interval),1);
                    C3_C3 = ones(length(C3_Interval),1);
                    Gam = ones(length(gamma_Interval),1);

                    for ith_C3 = 1:length(C3_Interval) 
                            C3 = C3_Interval(ith_C3);    % lambda3
                            C4 = C3;
                            
                            Best_Acc = 0;%必须要清0，设为0后，其余参数就不需要再重置了。
                            
                            for ith_lambda1 = 1:length(lambda1_Interval)    % lambda1
                                lambda1 = lambda1_Interval(ith_lambda1);    % lambda1
                                lambda3 = lambda1;
                                for ith_lambda2 = 1:length(lambda2_Interval)
                                  lambda2 = lambda2_Interval(ith_lambda2);
                                  lambda4 = lambda2;


                                          

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


                                                            C_s.C3 = C3;
                                                            C_s.C4 = C4;                                  


                                                            Outs_Train = Train_TLDM(Samples_A, Labels_A, Samples_B,Labels_B, Samples_Train, lambda1,lambda2 , C_s, Kernel, QPPs_Solver);

                                                            Acc = Predict_TLDM(Outs_Train, Samples_Predict,Labels_Predict, Samples_Train);   


                                                            Stop_Num = Stop_Num - 1;

                                                            disp([num2str(Stop_Num), ' step(s) remaining.'])


                                                            if Acc>Best_Acc
                                                                Best_Acc = Acc;
                                                                Best_lambda1 = lambda1;
                                                                Best_lambda2 = lambda2;
                                                                Best_C3 = C3;                      
                                                                Best_Kernel = Kernel;
                                                            end

                                                        end  % gamma
                                          

                                end
                            end    % lambda1
                            C3_ACC(ith_C3) = Best_Acc;
                            C3_lambda1(ith_C3) = Best_lambda1;
                            C3_lambda2(ith_C3) = Best_lambda2;
                            C3_C3(ith_C3) = Best_C3;
                            Gam(ith_C3) = Best_Kernel.gamma;
                            
                    end
                    
                    [Best_Acc,i]=max(C3_ACC);
                    
                    BestC_s.C3 = C3_C3(i);
                    BestC_s.C4 = C3_C3(i);
                    Best_lambda1 = C3_lambda1(i);
                    Best_lambda2 = C3_lambda2(i);
                    Best_Kernel.gamma = Gam(i);





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
%                 fprintf('%s\n', 'The Best_C3 is: ',num2str(BestC_s.C3));
%                 fprintf('%s\n','The Best_C4 is:',num2str(Best_C4));
            
                temp = [mean(t_Train),mean(100*Acc_Predict),...
                   Best_Kernel.gamma, Best_lambda1,Best_lambda2,BestC_s.C3];
                xlswrite('result.xlsx',temp,['AQ',num2str(iData+2),':AV',num2str(iData+2)]);
                xlswrite('result.xlsx',C3_ACC,['C',num2str(iData+77),':O',num2str(iData+77)]);
                
            case 3  %%TBSVM and FTBSVM
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

                    Ctrain1=awgn(Ctrain,0.1); % 高斯噪声
                    Ctest1=awgn(Ctest,0.1); % 高斯噪声
                    Ctrain= svdatanorm(Ctrain,'svpline');
                    Ctest= svdatanorm(Ctest,'scpline');
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
                    
                    
                    C3_ACC = ones(1,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                    C3_C1 = ones(length(C1_Interval),1);
                    C3_C3 = ones(length(C3_Interval),1);
                    Gam = ones(length(gamma_Interval),1);
                    
                    C3_ACC_ = ones(1,length(C3_Interval));%由于xlwrite是写行向量，所以设置为行向量
                    C3_C1_ = ones(length(C1_Interval),1);
                    C3_C3_ = ones(length(C3_Interval),1);
                    Gam_ = ones(length(gamma_Interval),1);


                    for ith_C3 = 1:length(C3_Interval) 
                              C3 = C3_Interval(ith_C3);    % lambda3
                              C4 = C3;
                              
                              Best_Acc = 0;%必须要清0，设为0后，其余参数就不需要再重置了。
                              Best_Acc_ = 0;
                              
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


                                            [tbsvm_Substruct] = tbsvmtrain(Samples_Train,Labels_Train,Parameter);
                                            [tbsvm_Substruct_] = IFtbsvmtrain_wang19(Samples_Train,Labels_Train,Parameter_);

                                            [Acc]= tbsvmclass(tbsvm_Substruct,Samples_Predict,Labels_Predict);  
                                            [Acc_]= tbsvmclass(tbsvm_Substruct_,Samples_Predict,Labels_Predict);   



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
                    end
                    
                    [Best_Acc,i]=max(C3_ACC);
                    [Best_Acc_,j]=max(C3_ACC_);
                    



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


                    tic         
                    [tbsvm_struct] = tbsvmtrain(Samples_Train,Labels_Train,Best_Parameter);
                    t_Train(Times) = toc;
                    
                    tic         
                    [tbsvm_struct_] = IFtbsvmtrain_wang19(Samples_Train,Labels_Train,Best_Parameter_);
                    t_Train_(Times) = toc;


                   % Predict the data
                    Acc_Predict(Times) = Best_Acc;
                    Acc_Predict_(Times) = Best_Acc_;

                end


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
                xlswrite('result.xlsx',temp,['AX',num2str(iData+2),':BB',num2str(iData+2)]);  
                xlswrite('result.xlsx',C3_ACC,['C',num2str(iData+152),':O',num2str(iData+152)]);
                
                temp_ = [mean(t_Train_),mean(Acc_Predict_),...
                   Best_Parameter_.p1,Best_Parameter_.CR,Best_Parameter_.CC];
                xlswrite('result.xlsx',temp_,['BD',num2str(iData+2),':BH',num2str(iData+2)]);  
                xlswrite('result.xlsx',C3_ACC_,['C',num2str(iData+177),':O',num2str(iData+177)]);
                
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


                %%%%%%%---------------------Save the statiatics---------------------%%%%%%%
% %                 Name = Str_Name(10:end-4);
%                 %                 Loc_Nam = [Location, '\', Name, '.txt'];
%                 %                 f = fopen(Loc_Nam, 'wt');
%                 disp('FTBLDM');
%                 fprintf('%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train)), '.']);
%                 fprintf('%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(100*Acc_Predict)) '%.']);
% %                 fprintf('%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(100*Acc_Predict)), '.']);
% %                 fprintf('%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
% %                 fprintf('%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);
% 
%                 fprintf('%s\n', 'The Best_gamma is: ',num2str(Best_Kernel.gamma));
% %                 fprintf('%s\n', 'The Best_lambda1 is: ',num2str(Best_lambda1));
% %                 fprintf('%s\n','The Best_lambda2 is:',num2str(Best_lambda2));
% %                 fprintf('%s\n', 'The Best_lambda3 is: ',num2str(Best_lambda3));
% %                 fprintf('%s\n','The Best_lambda4 is:',num2str(Best_lambda4));
% 
%                 fprintf('%s\n', 'The Best_C1 is: ',num2str(Best_C1));
%                 fprintf('%s\n', 'The Best_C2 is: ',num2str(Best_C2));
% %                 fprintf('%s\n', 'The Best_C3 is: ',num2str(Best_C3));
% %                 fprintf('%s\n','The Best_C4 is:',num2str(Best_C4));
% 
%                 fprintf('%s\n', ['The average training time is: ', sprintf('%.4f', mean(t_Train_)), '.']);
%                 fprintf('%s\n', ['The average predicting accurate is: ', sprintf('%2.2f', mean(100*Acc_Predict_)) '%.']);
% %                 fprintf('%s\n', ['The std predicting accurate is: ', sprintf('%.2f', std(100*Acc_Predict)), '.']);
% %                 fprintf('%s\n', ['The Margin MEAN is: ', sprintf('%0.2e', mean(MarginMEAN_Train)), '.']);
% %                 fprintf('%s\n', ['The Margin VARIANCE is: ', sprintf('%0.2e', mean(MarginSTD_Train)), '.']);
% 
%                 disp('TBLDM');
%                 fprintf('%s\n', 'The Best_gamma is: ',num2str(Best_Kernel_.gamma));
% %                 fprintf('%s\n', 'The Best_lambda1 is: ',num2str(Best_lambda1_));
% %                 fprintf('%s\n','The Best_lambda2 is:',num2str(Best_lambda2_));
% %                 fprintf('%s\n', 'The Best_lambda3 is: ',num2str(Best_lambda3_));
% %                 fprintf('%s\n','The Best_lambda4 is:',num2str(Best_lambda4_));
% 
%                 fprintf('%s\n', 'The Best_C1 is: ',num2str(Best_C1_));
%                 fprintf('%s\n', 'The Best_C2 is: ',num2str(Best_C2_));
% %                 fprintf('%s\n', 'The Best_C3 is: ',num2str(Best_C3_));
% %                 fprintf('%s\n','The Best_C4 is:',num2str(Best_C4_));                

                temp = [mean(t_Train),mean(100*Acc_Predict),...
                   Best_Kernel.gamma,BestC_s.C1];
                xlswrite('result.xlsx',temp,['AE',num2str(iData+6),':AH',num2str(iData+6)]);
                xlswrite('result.xlsx',C3_ACC,['C',num2str(iData+102),':O',num2str(iData+102)]);

                temp_ = [mean(t_Train_),mean(100*Acc_Predict_),...
                   Best_Kernel_.gamma,BestC_s_.C1];
                xlswrite('result.xlsx',temp_,['AK',num2str(iData+6),':AN',num2str(iData+6)]);
                xlswrite('result.xlsx',C3_ACC_,['C',num2str(iData+127),':O',num2str(iData+127)]);
        end
    end
end   
