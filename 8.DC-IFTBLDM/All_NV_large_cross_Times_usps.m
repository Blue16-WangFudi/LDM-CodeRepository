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
% lambda1_Interval = 2.^(-8:-7);
% lambda2_Interval = 2.^(-8:-7);
% C1_Interval = 2.^(-8:-7);
% C3_Interval = 2.^(-8:-7);


Best_u = 0.4;


%% Counts
N_Times = 1;
K_fold = 5;
TrainRate = 0.3;
switch Kernel.Type
    case 'Linear'
        Stop_Num = 7*N_Times*length(lambda1_Interval)*length(lambda2_Interval)*length(C1_Interval)*length(C3_Interval) + 1;
    case 'RBF'
        Stop_Num = 5*N_Times*K_fold*length(lambda1_Interval)*length(lambda2_Interval)*length(C1_Interval)*length(C3_Interval)*length(gamma_Interval) + 1;
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
for iData =22:26
%                 Str_Name = Str(iData).Name;
%                 Output = load(Str_Name);
%                 Data_Name = fieldnames(Output);   % A struct data
%                 Data_Original = getfield(Output, Data_Name{1}); % Abstract the data
%                 % Normalization
%                 Data_Original = [mapminmax(Data_Original(:, 1:end-1)', 0, 1)', Data_Original(:, end)]; % Map the original data to value between [0, 1] by colum
% %                 M_Original = size(Data_Original, 1);
% %                 Data_Original = Data_Original(randperm(M_Original), :);


    if (iData==1)
        load('Data_mat_n\Monk_1.mat');
        X = Monk_1;
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);     
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
        
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');
   
    end

    if(iData==2)
        load('Data_mat_n\Monk_2.mat');
        X = Monk_2;    
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);     
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
        
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');
    end

    if (iData==3)
        load('Data_mat_n\Monk_3.mat');
        X = Monk_3;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
        
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');
    end
    
     if (iData==4)
        load( 'Data_mat_x\tic_tac_toe.mat');
        X = tic_tac_toe;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);

%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');        

     end
     if (iData==5)
        load('Data_mat\Promoters.mat');
        X = Promoters;     
        
         % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);            
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
        
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');
     end
     
     if (iData==6)
        load('Data_mat\Iris.mat');
        X = Iris;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
        
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');
     end
      
     if (iData==7)
        load('Data_mat\Wine.mat');
        X = Wine;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
%         
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');        

     end
      
      if (iData==8)
        load( 'Data_mat\Haberman.mat');
        X = Haberman;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
        
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');       
        
      end
     
      if (iData==9)
        load('Data_mat_x\breast_cancer.mat');
        X = breast_cancer;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
        
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');        

      end
      
      if (iData==10)
        load('Data_mat_n\wdbc_data.mat')
        load('Data_mat_n\wdbc_label.mat')
        X = [wdbc_data,wdbc_label];
        X(find(X(:,end)==2),end)=-1;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
        
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');        
     end

    if (iData==11)
        load('Data_mat_n\Wpbc.mat');
        X = Wpbc;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
        
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');        

    end
    
    if (iData==12)
        load( 'Data_mat\Breast_cancer_Original.mat');
        X = Breast_cancer_Original;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
        
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');        

    end
    
     if (iData==13)
        load('Data_mat\New_thyroid.mat');
        X = New_thyroid;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
        
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');        

     end
      
      if (iData==14)
        load('Data_mat_n\echocardiogram_data.mat');
        load('Data_mat_n\echocardiogram_label.mat');
        X = [x,y];
        X(find(X(:,end)==0),end)=-1;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);  
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
        
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');        
      end
    
      if (iData==15)
        load('Data_mat_x\heart_statlog.mat');
        X = heart_statlog;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
        
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');        

     end
     
     if (iData==16)
        load( 'Data_mat\Hepatitis.mat');
        X = Hepatitis;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
        
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');        

     end
     
     if (iData==17)
        load('Data_mat\Sonar.mat');
        X = Sonar;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
%         
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');        

     end
    
     if (iData==18)
        load( 'Data_mat\Glass.mat');
        X = Glass;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);            
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);

%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');        
        
     end

    if(iData==19)
        load('Data_mat_n\plrx.txt');
        plrx(find(plrx(:,end)==2),end)=-1;
        X = plrx;
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);

%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');
        
        
    end
    
     if (iData==20)
        load('Data_mat_x\credit_a.mat');
        X = credit_a;        
        
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
%         Indices = crossvalind('Kfold', length(dtrain),K_fold);
        
%         Ctrain= svdatanorm(Ctrain,'svpline');
%         Ctest= svdatanorm(Ctest,'scpline');
     end

    if(iData==21)
        load('Data_mat\CMC.mat');
        X= CMC;
        Y= CMC(:,end);
        X(find(X(:,end)~=1),end)=-1;
        % Normalization
        X = [mapminmax(X(:, 1:end-1)', 0, 1)', X(:, end)]; % Map the original data to value between [0, 1] by colum
        M_Original = size(X, 1);
        X = X(randperm(M_Original), :);             
        
%         [test, train] = Data_Rate(X, TrainRate);        
%         Ctrain=train(:,1:end-1);
%         dtrain= train(:,end);
%         Ctest= test(:,1:end-1);
%         dtest= test(:,end);
    end
    
    
    
    if(iData==22)
        load('Data_mat\usps\TFS_data\train_0');
        load('Data_mat\usps\TFS_data\test_0');
        
        train_0 = [mapminmax(train_0(:, 1:end-1)', 0, 1)', train_0(:, end)];
        M_Original = size(train_0, 1);
        train_0 = train_0(randperm(M_Original), :);
        
        test_0 = [mapminmax(test_0(:, 1:end-1)', 0, 1)', test_0(:, end)];
        M_Original = size(test_0, 1);
        test_0 = test_0(randperm(M_Original), :);
        
        Ctrain= train_0(:,1:end-1);
        dtrain=  train_0(:,end);
        Ctest= test_0(:,1:end-1);
        dtest=  test_0(:,end);
    end

    if(iData==23)
        load('Data_mat\usps\TFS_data\train_2');
        load('Data_mat\usps\TFS_data\test_2');
        
        train_2 = [mapminmax(train_2(:, 1:end-1)', 0, 1)', train_2(:, end)];
        M_Original = size(train_2, 1);
        train_2 = train_2(randperm(M_Original), :);
        
        test_2 = [mapminmax(test_2(:, 1:end-1)', 0, 1)', test_2(:, end)];
        M_Original = size(test_2, 1);
        test_2 = test_2(randperm(M_Original), :);
        
        Ctrain= train_2(:,1:end-1);
        dtrain=  train_2(:,end);
        Ctest= test_2(:,1:end-1);
        dtest=  test_2(:,end);
    end

    if(iData==24)
        load('Data_mat\usps\TFS_data\train_4');
        load('Data_mat\usps\TFS_data\test_4');
        
        train_4 = [mapminmax(train_4(:, 1:end-1)', 0, 1)', train_4(:, end)];
        M_Original = size(train_4, 1);
        train_4 = train_4(randperm(M_Original), :);
        
        test_4 = [mapminmax(test_4(:, 1:end-1)', 0, 1)', test_4(:, end)];
        M_Original = size(test_4, 1);
        test_4 = test_4(randperm(M_Original), :);
        
        Ctrain= train_4(:,1:end-1);
        dtrain=  train_4(:,end);
        Ctest= test_4(:,1:end-1);
        dtest=  test_4(:,end);
    end

    if(iData==25)
        load('Data_mat\usps\TFS_data\train_6');
        load('Data_mat\usps\TFS_data\test_6');
        
        train_6 = [mapminmax(train_6(:, 1:end-1)', 0, 1)', train_6(:, end)];
        M_Original = size(train_6, 1);
        train_6 = train_6(randperm(M_Original), :);
        
        test_6 = [mapminmax(test_6(:, 1:end-1)', 0, 1)', test_6(:, end)];
        M_Original = size(test_6, 1);
        test_6 = test_6(randperm(M_Original), :);
        
        Ctrain= train_6(:,1:end-1);
        dtrain=  train_6(:,end);
        Ctest= test_6(:,1:end-1);
        dtest=  test_6(:,end);
    end
    
    if(iData==26)
        load('Data_mat\usps\TFS_data\train_8');
        load('Data_mat\usps\TFS_data\test_8');
        
        train_8 = [mapminmax(train_8(:, 1:end-1)', 0, 1)', train_8(:, end)];
        M_Original = size(train_8, 1);
        train_8 = train_8(randperm(M_Original), :);
        
        test_8 = [mapminmax(test_8(:, 1:end-1)', 0, 1)', test_8(:, end)];
        M_Original = size(test_8, 1);
        test_8 = test_8(randperm(M_Original), :);
        
        Ctrain= train_8(:,1:end-1);
        dtrain=  train_8(:,end);
        Ctest= test_8(:,1:end-1);
        dtest=  test_8(:,end);
    end
    
    

    
      
    for type=1:1
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
                
                for Times = 1: N_Times
                    
                    
%                     [test, train] = Data_Rate(X, TrainRate); 


%                     Ctrain=train(:,1:end-1);
%                     dtrain= train(:,end);
%                     Ctest= test(:,1:end-1);
%                     dtest= test(:,end);

                    Samples_Train = Ctrain;
                    Labels_Train = dtrain;                     
                    A_Samples_Train = Samples_Train(Labels_Train == 1,:);
                    A_Labels_Train = Labels_Train(Labels_Train == 1,:);
                    
                    B_Samples_Train = Samples_Train(Labels_Train == -1,:);
                    B_Labels_Train = Labels_Train(Labels_Train == -1,:);
                    
                    Samples_Predict = Ctest;
                   
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
                                                    
                                                    Indices1 = crossvalind('Kfold', length(A_Labels_Train),K_fold);
                                                    Indices2 = crossvalind('Kfold', length(B_Labels_Train),K_fold);
                                                    %分层交叉验证
                                                                                                        
                                                    Acc_SubPredict = zeros(K_fold, 1);
                                                    for repeat = 1:K_fold
                                                            I_SubTrain1 = ~(Indices1 == repeat);
                                                            I_SubTrain2 = ~(Indices2 == repeat);
                                                            
                                                            
                                                            A_Samples_SubTrain = A_Samples_Train(I_SubTrain1,:);
                                                            A_Labels_SubTrain = A_Labels_Train(I_SubTrain1,:);
                                                            B_Samples_SubTrain = B_Samples_Train(I_SubTrain2,:);
                                                            B_Labels_SubTrain = B_Labels_Train(I_SubTrain2,:); 
                                                            
                                                            Samples_SubTrain = [A_Samples_SubTrain;B_Samples_SubTrain];
                                                            Labels_SubTrain = [A_Labels_SubTrain;B_Labels_SubTrain];
                                                                                                                                                                                   
                                                            I_SubPredict1 = ~I_SubTrain1;
                                                            I_SubPredict2 = ~I_SubTrain2;
                                                            
                                                            A_Samples_SubPredict = A_Samples_Train(I_SubPredict1,:);
                                                            A_Labels_SubPredict = A_Labels_Train(I_SubPredict1,:);
                                                            B_Samples_SubPredict = B_Samples_Train(I_SubPredict2,:);
                                                            B_Labels_SubPredict = B_Labels_Train(I_SubPredict2,:);
                                                            
                                                            Samples_SubPredict = [A_Samples_SubPredict;B_Samples_SubPredict];
                                                            Labels_SubPredict = [A_Labels_SubPredict;B_Labels_SubPredict];
                                                            

                                                            
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
                            Bests = IFuzzy_MemberShip(Samples_Train, Labels_Train, Best_Kernel, Best_u); 
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
                
                xlswrite('result.xlsx',100*(C3_ACCp(1,:)),['C',num2str(iData+33),':O',num2str(iData+33)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(2,:)),['Q',num2str(iData+33),':AC',num2str(iData+33)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(3,:)),['AE',num2str(iData+33),':AQ',num2str(iData+33)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(4,:)),['AS',num2str(iData+33),':BE',num2str(iData+33)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(5,:)),['BG',num2str(iData+33),':BS',num2str(iData+33)]);
%                 xlswrite('result.xlsx',100*mean(C3_ACCp),['BU',num2str(iData+33),':CG',num2str(iData+33)]);
%                 
%                 
%                 xlswrite('result.xlsx',100*Acc_Predict',['CI',num2str(iData+33),':CM',num2str(iData+33)]);%xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的

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
                    
%                     [test, train] = Data_Rate(X, TrainRate);        
%                     Ctrain=train(:,1:end-1);
%                     dtrain= train(:,end);
%                     Ctest= test(:,1:end-1);
%                     dtest= test(:,end);

%                     Ctrain1=awgn(Ctrain,0.1); % 高斯噪声
%                     Ctest1=awgn(Ctest,0.1); % 高斯噪声
%                     Ctrain= svdatanorm(Ctrain,'svpline');
%                     Ctest= svdatanorm(Ctest,'scpline');

%分层交叉验证，划分正负集
                    Samples_Train = Ctrain;
                    Labels_Train = dtrain;                     
                    A_Samples_Train = Samples_Train(Labels_Train == 1,:);
                    A_Labels_Train = Labels_Train(Labels_Train == 1,:);
                    
                    B_Samples_Train = Samples_Train(Labels_Train == -1,:);
                    B_Labels_Train = Labels_Train(Labels_Train == -1,:);
                    
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
                                      
                                        Indices1 = crossvalind('Kfold', length(A_Labels_Train),K_fold);
                                        Indices2 = crossvalind('Kfold', length(B_Labels_Train),K_fold);
                                        %分层交叉验证
                                        
                                        
                                        Acc_SubPredict = zeros(K_fold, 1);
                                        Acc_SubPredict_ = zeros(K_fold, 1);
                                        Acc_SubPredict_1 = zeros(K_fold, 1);                                      
                                       for repeat = 1:K_fold
                                           
                                            I_SubTrain1 = ~(Indices1 == repeat);
                                            I_SubTrain2 = ~(Indices2 == repeat);


                                            A_Samples_SubTrain = A_Samples_Train(I_SubTrain1,:);
                                            A_Labels_SubTrain = A_Labels_Train(I_SubTrain1,:);
                                            B_Samples_SubTrain = B_Samples_Train(I_SubTrain2,:);
                                            B_Labels_SubTrain = B_Labels_Train(I_SubTrain2,:); 

                                            Samples_SubTrain = [A_Samples_SubTrain;B_Samples_SubTrain];
                                            Labels_SubTrain = [A_Labels_SubTrain;B_Labels_SubTrain];

                                            I_SubPredict1 = ~I_SubTrain1;
                                            I_SubPredict2 = ~I_SubTrain2;

                                            A_Samples_SubPredict = A_Samples_Train(I_SubPredict1,:);
                                            A_Labels_SubPredict = A_Labels_Train(I_SubPredict1,:);
                                            B_Samples_SubPredict = B_Samples_Train(I_SubPredict2,:);
                                            B_Labels_SubPredict = B_Labels_Train(I_SubPredict2,:);

                                            Samples_SubPredict = [A_Samples_SubPredict;B_Samples_SubPredict];
                                            Labels_SubPredict = [A_Labels_SubPredict;B_Labels_SubPredict];                                    




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
                
                xlswrite('result.xlsx',(C3_ACCp(1,:)),['C',num2str(iData+123),':O',num2str(iData+123)]);
%                 xlswrite('result.xlsx',(C3_ACCp(2,:)),['Q',num2str(iData+123),':AC',num2str(iData+123)]);
%                 xlswrite('result.xlsx',(C3_ACCp(3,:)),['AE',num2str(iData+123),':AQ',num2str(iData+123)]);
%                 xlswrite('result.xlsx',(C3_ACCp(4,:)),['AS',num2str(iData+123),':BE',num2str(iData+123)]);
%                 xlswrite('result.xlsx',(C3_ACCp(5,:)),['BG',num2str(iData+123),':BS',num2str(iData+123)]);
%                 xlswrite('result.xlsx',mean(C3_ACCp),['BU',num2str(iData+123),':CG',num2str(iData+123)]);
%                 
%                 xlswrite('result.xlsx',Acc_Predict',['CI',num2str(iData+123),':CM',num2str(iData+123)]);  %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的              
                
                
                temp_ = [mean(t_Train_Predict_),mean(Acc_Predict_),std(Acc_Predict_),Acc_Leader_,...
                   Gam_(Times_Leader_,j_Leader),C3_C1_(Times_Leader_,j_Leader),C3_C3_(Times_Leader_,j_Leader)];
                xlswrite('result.xlsx',temp_,['M',num2str(iData+2),':S',num2str(iData+2)]);  
                
                xlswrite('result.xlsx',(C3_ACCp_(1,:)),['C',num2str(iData+63),':O',num2str(iData+63)]);
%                 xlswrite('result.xlsx',(C3_ACCp_(2,:)),['Q',num2str(iData+63),':AC',num2str(iData+63)]);
%                 xlswrite('result.xlsx',(C3_ACCp_(3,:)),['AE',num2str(iData+63),':AQ',num2str(iData+63)]);
%                 xlswrite('result.xlsx',(C3_ACCp_(4,:)),['AS',num2str(iData+63),':BE',num2str(iData+63)]);
%                 xlswrite('result.xlsx',(C3_ACCp_(5,:)),['BG',num2str(iData+63),':BS',num2str(iData+63)]);
%                 xlswrite('result.xlsx',mean(C3_ACCp_),['BU',num2str(iData+63),':CG',num2str(iData+63)]);
%                 
%                 xlswrite('result.xlsx',Acc_Predict_',['CI',num2str(iData+63),':CM',num2str(iData+63)]);    %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的
                
                temp_1 = [mean(t_Train_Predict_1),mean(Acc_Predict_1),std(Acc_Predict_1),Acc_Leader_1,...
                   Gam_1(Times_Leader_1,k_Leader),C3_C1_1(Times_Leader_1,k_Leader),C3_C3_1(Times_Leader_1,k_Leader)];
                xlswrite('result.xlsx',temp_1,['U',num2str(iData+2),':AA',num2str(iData+2)]);  
                
                xlswrite('result.xlsx',(C3_ACCp_1(1,:)),['C',num2str(iData+93),':O',num2str(iData+93)]);
%                 xlswrite('result.xlsx',(C3_ACCp_1(2,:)),['Q',num2str(iData+93),':AC',num2str(iData+93)]);
%                 xlswrite('result.xlsx',(C3_ACCp_1(3,:)),['AE',num2str(iData+93),':AQ',num2str(iData+93)]);
%                 xlswrite('result.xlsx',(C3_ACCp_1(4,:)),['AS',num2str(iData+93),':BE',num2str(iData+93)]);
%                 xlswrite('result.xlsx',(C3_ACCp_1(5,:)),['BG',num2str(iData+93),':BS',num2str(iData+93)]);
%                 xlswrite('result.xlsx',mean(C3_ACCp_1),['BU',num2str(iData+93),':CG',num2str(iData+93)]);
%                 
%                 xlswrite('result.xlsx',Acc_Predict_1',['CI',num2str(iData+93),':CM',num2str(iData+93)]);    %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的                
%                 
                
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
                    
%                     [test, train] = Data_Rate(X, TrainRate);        
%                     Ctrain=train(:,1:end-1);
%                     dtrain= train(:,end);
%                     Ctest= test(:,1:end-1);
%                     dtest= test(:,end);
%                     Ctrain=awgn(Ctrain,0.1); % 高斯噪声
%                     Ctest=awgn(Ctest,0.1); % 高斯噪声
%                     Ctrain= svdatanorm(Ctrain,'svpline');
%                     Ctest= svdatanorm(Ctest,'scpline');
                    %分层交叉验证，划分正负集
                    Samples_Train = Ctrain;
                    Labels_Train = dtrain;                     
                    A_Samples_Train = Samples_Train(Labels_Train == 1,:);
                    A_Labels_Train = Labels_Train(Labels_Train == 1,:);
                    
                    B_Samples_Train = Samples_Train(Labels_Train == -1,:);
                    B_Labels_Train = Labels_Train(Labels_Train == -1,:);
                    
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
                                     
                                    Indices1 = crossvalind('Kfold', length(A_Labels_Train),K_fold);
                                    Indices2 = crossvalind('Kfold', length(B_Labels_Train),K_fold);
                                    %分层交叉验证
                                    
                                    
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
                                        
                                        I_SubTrain1 = ~(Indices1 == repeat);
                                        I_SubTrain2 = ~(Indices2 == repeat);


                                        A_Samples_SubTrain = A_Samples_Train(I_SubTrain1,:);
                                        A_Labels_SubTrain = A_Labels_Train(I_SubTrain1,:);
                                        B_Samples_SubTrain = B_Samples_Train(I_SubTrain2,:);
                                        B_Labels_SubTrain = B_Labels_Train(I_SubTrain2,:); 

                                        Samples_SubTrain = [A_Samples_SubTrain;B_Samples_SubTrain];
                                        Labels_SubTrain = [A_Labels_SubTrain;B_Labels_SubTrain];

                                        I_SubPredict1 = ~I_SubTrain1;
                                        I_SubPredict2 = ~I_SubTrain2;

                                        A_Samples_SubPredict = A_Samples_Train(I_SubPredict1,:);
                                        A_Labels_SubPredict = A_Labels_Train(I_SubPredict1,:);
                                        B_Samples_SubPredict = B_Samples_Train(I_SubPredict2,:);
                                        B_Labels_SubPredict = B_Labels_Train(I_SubPredict2,:);

                                        Samples_SubPredict = [A_Samples_SubPredict;B_Samples_SubPredict];
                                        Labels_SubPredict = [A_Labels_SubPredict;B_Labels_SubPredict];


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
                                        s_ = Fuzzy_MemberShip(Samples_SubTrain, Labels_SubTrain, Kernel, Best_u);
                                        
                                        C_s_.s1 = s_.s1;
                                        C_s_.s2 = s_.s2;



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
                            Bests_ = Fuzzy_MemberShip(Samples_Train, Labels_Train, Best_Kernel_, Best_u); 
                            BestC_s_.s1 = Bests_.s1;
                            BestC_s_.s2 = Bests_.s2;

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
                
                xlswrite('result.xlsx',100*(C3_ACCp(1,:)),['C',num2str(iData+153),':O',num2str(iData+153)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(2,:)),['Q',num2str(iData+153),':AC',num2str(iData+153)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(3,:)),['AE',num2str(iData+153),':AQ',num2str(iData+153)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(4,:)),['AS',num2str(iData+153),':BE',num2str(iData+153)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(5,:)),['BG',num2str(iData+153),':BS',num2str(iData+153)]);
%                 xlswrite('result.xlsx',100*mean(C3_ACCp),['BU',num2str(iData+153),':CG',num2str(iData+153)]);
%                 
%                 xlswrite('result.xlsx',100*Acc_Predict',['CI',num2str(iData+153),':CM',num2str(iData+153)]);  %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的                

                temp_ = [mean(t_Train_Predict_),100*mean(Acc_Predict_),100*std(Acc_Predict_),100*Acc_Leader_,...
                   Gam(Times_Leader_,j_Leader),C3_C1(Times_Leader_,j_Leader)];
                xlswrite('result.xlsx',temp_,['AR',num2str(iData+2),':AW',num2str(iData+2)]);
                
                xlswrite('result.xlsx',100*(C3_ACCp_(1,:)),['C',num2str(iData+183),':O',num2str(iData+183)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp_(2,:)),['Q',num2str(iData+183),':AC',num2str(iData+183)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp_(3,:)),['AE',num2str(iData+183),':AQ',num2str(iData+183)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp_(4,:)),['AS',num2str(iData+183),':BE',num2str(iData+183)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp_(5,:)),['BG',num2str(iData+183),':BS',num2str(iData+183)]);
%                 xlswrite('result.xlsx',100*mean(C3_ACCp_),['BU',num2str(iData+183),':CG',num2str(iData+183)]);
%                 
%                 xlswrite('result.xlsx',100*Acc_Predict_',['CI',num2str(iData+183),':CM',num2str(iData+183)]);  %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的                
%                 
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
                for Times = 1: N_Times
                    
%                     [test, train] = Data_Rate(X, TrainRate);        
%                     Ctrain=train(:,1:end-1);
%                     dtrain= train(:,end);
%                     Ctest= test(:,1:end-1);
%                     dtest= test(:,end);
                    

%                     Ctrain=awgn(Ctrain,0.1); % 高斯噪声
%                     Ctest=awgn(Ctest,0.1); % 高斯噪声
%                     Ctrain= svdatanorm(Ctrain,'svpline');
%                     Ctest= svdatanorm(Ctest,'scpline');
                    %分层交叉验证，划分正负集
                    Samples_Train = Ctrain;
                    Labels_Train = dtrain;                     
                    A_Samples_Train = Samples_Train(Labels_Train == 1,:);
                    A_Labels_Train = Labels_Train(Labels_Train == 1,:);
                    
                    B_Samples_Train = Samples_Train(Labels_Train == -1,:);
                    B_Labels_Train = Labels_Train(Labels_Train == -1,:);
                    
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
                                        
                                        
                                        Indices1 = crossvalind('Kfold', length(A_Labels_Train),K_fold);
                                        Indices2 = crossvalind('Kfold', length(B_Labels_Train),K_fold);
                                        %分层交叉验证
                                        
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
                                        
                                        I_SubTrain1 = ~(Indices1 == repeat);
                                        I_SubTrain2 = ~(Indices2 == repeat);


                                        A_Samples_SubTrain = A_Samples_Train(I_SubTrain1,:);
                                        A_Labels_SubTrain = A_Labels_Train(I_SubTrain1,:);
                                        B_Samples_SubTrain = B_Samples_Train(I_SubTrain2,:);
                                        B_Labels_SubTrain = B_Labels_Train(I_SubTrain2,:); 

                                        Samples_SubTrain = [A_Samples_SubTrain;B_Samples_SubTrain];
                                        Labels_SubTrain = [A_Labels_SubTrain;B_Labels_SubTrain];

                                        I_SubPredict1 = ~I_SubTrain1;
                                        I_SubPredict2 = ~I_SubTrain2;

                                        A_Samples_SubPredict = A_Samples_Train(I_SubPredict1,:);
                                        A_Labels_SubPredict = A_Labels_Train(I_SubPredict1,:);
                                        B_Samples_SubPredict = B_Samples_Train(I_SubPredict2,:);
                                        B_Labels_SubPredict = B_Labels_Train(I_SubPredict2,:);

                                        Samples_SubPredict = [A_Samples_SubPredict;B_Samples_SubPredict];
                                        Labels_SubPredict = [A_Labels_SubPredict;B_Labels_SubPredict];

                                        
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
                                    Bests_ = DC_Fuzzy_MemberShip(Samples_Train, Labels_Train, Best_Kernel, Best_u); 


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
                    Bests_ = DC_Fuzzy_MemberShip(Samples_Train, Labels_Train, Best_Kernel, Best_u); 
                    
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
                
                xlswrite('result.xlsx',100*(C3_ACCp(1,:)),['C',num2str(iData+213),':O',num2str(iData+213)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(2,:)),['Q',num2str(iData+213),':AC',num2str(iData+213)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(3,:)),['AE',num2str(iData+213),':AQ',num2str(iData+213)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(4,:)),['AS',num2str(iData+213),':BE',num2str(iData+213)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(5,:)),['BG',num2str(iData+213),':BS',num2str(iData+213)]);
%                 xlswrite('result.xlsx',100*mean(C3_ACCp),['BU',num2str(iData+213),':CG',num2str(iData+213)]);
%                 
%                 xlswrite('result.xlsx',100*Acc_Predict',['CI',num2str(iData+213),':CM',num2str(iData+213)]);  %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的                
                
                temp = [mean(t_Train_Predict_),100*mean(Acc_Predict_),100*std(Acc_Predict_),100*Acc_Leader_,...
                   Gam_(Times,j_Leader),C3_C1_(Times,j_Leader)];
                xlswrite('result.xlsx',temp,['BF',num2str(iData+2),':BK',num2str(iData+2)]);
                
                xlswrite('result.xlsx',100*(C3_ACCp_(1,:)),['C',num2str(iData+243),':O',num2str(iData+243)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp_(2,:)),['Q',num2str(iData+243),':AC',num2str(iData+243)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp_(3,:)),['AE',num2str(iData+243),':AQ',num2str(iData+243)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp_(4,:)),['AS',num2str(iData+243),':BE',num2str(iData+243)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp_(5,:)),['BG',num2str(iData+243),':BS',num2str(iData+243)]);
%                 xlswrite('result.xlsx',100*mean(C3_ACCp_),['BU',num2str(iData+243),':CG',num2str(iData+243)]);
%                 
%                 xlswrite('result.xlsx',100*Acc_Predict_',['CI',num2str(iData+243),':CM',num2str(iData+243)]);    %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的             
                
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

%                     [test, train] = Data_Rate(X, TrainRate);        
%                     Ctrain=train(:,1:end-1);
%                     dtrain= train(:,end);
%                     Ctest= test(:,1:end-1);
%                     dtest= test(:,end);                    
% 
%                     Ctrain1=awgn(Ctrain,0.1); % 高斯噪声
%                     Ctest1=awgn(Ctest,0.1); % 高斯噪声
%                     Ctrain= svdatanorm(Ctrain,'svpline');
%                     Ctest= svdatanorm(Ctest,'scpline');
                    %分层交叉验证，划分正负集
                    Samples_Train = Ctrain;
                    Labels_Train = dtrain;                     
                    A_Samples_Train = Samples_Train(Labels_Train == 1,:);
                    A_Labels_Train = Labels_Train(Labels_Train == 1,:);
                    
                    B_Samples_Train = Samples_Train(Labels_Train == -1,:);
                    B_Labels_Train = Labels_Train(Labels_Train == -1,:);
                    
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
                                                        Indices1 = crossvalind('Kfold', length(A_Labels_Train),K_fold);
                                                        Indices2 = crossvalind('Kfold', length(B_Labels_Train),K_fold);
                                                        %分层交叉验证

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
                                                            
                                                            I_SubTrain1 = ~(Indices1 == repeat);
                                                            I_SubTrain2 = ~(Indices2 == repeat);
                                                            
                                                            
                                                            A_Samples_SubTrain = A_Samples_Train(I_SubTrain1,:);
                                                            A_Labels_SubTrain = A_Labels_Train(I_SubTrain1,:);
                                                            B_Samples_SubTrain = B_Samples_Train(I_SubTrain2,:);
                                                            B_Labels_SubTrain = B_Labels_Train(I_SubTrain2,:); 
                                                            
                                                            Samples_SubTrain = [A_Samples_SubTrain;B_Samples_SubTrain];
                                                            Labels_SubTrain = [A_Labels_SubTrain;B_Labels_SubTrain];
                                                                                                                                                                                   
                                                            I_SubPredict1 = ~I_SubTrain1;
                                                            I_SubPredict2 = ~I_SubTrain2;
                                                            
                                                            A_Samples_SubPredict = A_Samples_Train(I_SubPredict1,:);
                                                            A_Labels_SubPredict = A_Labels_Train(I_SubPredict1,:);
                                                            B_Samples_SubPredict = B_Samples_Train(I_SubPredict2,:);
                                                            B_Labels_SubPredict = B_Labels_Train(I_SubPredict2,:);
                                                            
                                                            Samples_SubPredict = [A_Samples_SubPredict;B_Samples_SubPredict];
                                                            Labels_SubPredict = [A_Labels_SubPredict;B_Labels_SubPredict];
                                                            
                                                            C_s_.C = C1*abs(Labels_SubTrain);
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
                
                xlswrite('result.xlsx',100*(C3_ACCp(1,:)),['C',num2str(iData+273),':O',num2str(iData+273)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(2,:)),['Q',num2str(iData+273),':AC',num2str(iData+273)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(3,:)),['AE',num2str(iData+273),':AQ',num2str(iData+273)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(4,:)),['AS',num2str(iData+273),':BE',num2str(iData+273)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp(5,:)),['BG',num2str(iData+273),':BS',num2str(iData+273)]);
%                 xlswrite('result.xlsx',100*mean(C3_ACCp),['BU',num2str(iData+273),':CG',num2str(iData+273)]);
                
%                 xlswrite('result.xlsx',100*Acc_Predict',['CI',num2str(iData+273),':CM',num2str(iData+273)]); %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的                   
                
                temp_ = [mean(t_Train_Predict_),100*mean(Acc_Predict_),100*std(Acc_Predict_),100*Acc_Leader_,...
                    Gam_(Times_Leader_,j_Leader),C3_lambda1_(Times_Leader_,j_Leader),C3_lambda2_(Times_Leader_,j_Leader),C3_C1_(Times_Leader_,j_Leader)];
                xlswrite('result.xlsx',temp_,['BV',num2str(iData+2),':CC',num2str(iData+2)]);
                
                xlswrite('result.xlsx',100*(C3_ACCp_(1,:)),['C',num2str(iData+303),':O',num2str(iData+303)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp_(2,:)),['Q',num2str(iData+303),':AC',num2str(iData+303)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp_(3,:)),['AE',num2str(iData+303),':AQ',num2str(iData+303)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp_(4,:)),['AS',num2str(iData+303),':BE',num2str(iData+303)]);
%                 xlswrite('result.xlsx',100*(C3_ACCp_(5,:)),['BG',num2str(iData+303),':BS',num2str(iData+303)]);
%                 xlswrite('result.xlsx',100*mean(C3_ACCp_),['BU',num2str(iData+303),':CG',num2str(iData+303)]);
%                   
%                 xlswrite('result.xlsx',100*Acc_Predict_',['CI',num2str(iData+303),':CM',num2str(iData+303)]);     %xlswrite只能写入行向量，如果是列向量，则只写入第一个元素，所有都是相同的               
                
        end
    end
end   
