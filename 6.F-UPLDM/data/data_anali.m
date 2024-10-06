tic
close all;
clear all;
clc
for index=28:28
    %%
    if(index==2)
        load('Data/monks_2_train.txt');
        load('Data/monks_2_test.txt');
        Ctrain= monks_2_train(:,2:end);
        dtrain=  monks_2_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= monks_2_test(:,2:end);
        dtest=  monks_2_test(:,1);
        dtest(find(dtest==0))=-1;
        disp('Monk 2');
    end
    %%
    if (index==3)
        load('Data/monks_3_train.txt');
        load('Data/monks_3_test.txt');
        Ctrain= monks_3_train(:,2:end);
        dtrain=  monks_3_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= monks_3_test(:,2:end);
        dtest=  monks_3_test(:,1);
        dtest(find(dtest==0))=-1;
        disp('Monk 3');
    end
    %%
    if (index==4)
        load('Data/SPECT_train.txt')
        load('Data/SPECT_test.txt');
        Ctrain = SPECT_train(:,2:end);
        dtrain=  SPECT_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= SPECT_test(:,2:end);
        dtest=  SPECT_test(:,1);
        dtest(find(dtest==0))=-1;
        disp('Spect');
    end
    %%
    if (index==5)
        
        load('Data/Haberman_dataset.mat')
        X=Haberman_data(:,1:end-1);
        Y= Haberman_data(:,end);
        Y(find(Y==2))=-1;
        Ctrain=X(1:150,:);
        dtrain= Y(1:150,:);
        Ctest= X(151:end,:);
        dtest= Y(151:end,:);
        disp ('Heberman');
    end
    
    %%
    if (index==6)
        load('Data/heartdata.mat');
        X=heartdata(:,1:end-1);
        Y= heartdata(:,end);
        Y(find(Y==2))=-1;
        Ctrain=X(1:150,:);
        dtrain= Y(1:150,:);
        Ctest= X(151:end,:);
        dtest= Y(151:end,:);
        disp ('Statlog');
    end
    %%
    if (index==7)
        load('Data/ionosphere_data.mat');
        X=data(:,2:end);
        Y= data(:,1);
        Y(find(Y==0))=-1;
        Ctrain=X(1:200,:);
        dtrain= Y(1:200,:);
        Ctest= X(201:end,:);
        dtest= Y(201:end,:);
        disp('Ionosphere')
    end
    %%
    if (index==8)
        data= xlsread('Data/Pima-Indian.xlsx');
        X=data(:,2:end);
        Y= data(:,1);
        Y(find(Y==0))=-1;
        Ctrain=X(1:300,:);
        dtrain= Y(1:300,:);
        Ctest= X(301:end,:);
        dtest= Y(301:end,:);
        disp('Pima-Indian');
    end
    %%
    
    if (index==9)
        load('Data/wdbc_data.mat')
        load('Data/wdbc_label.mat')
        X=wdbc_data;
        Y= wdbc_label;
        Y(find(Y==2))=-1;
        Ctrain=X(1:400,:);
        dtrain= Y(1:400,:);
        Ctest= X(401:end,:);
        dtest= Y(401:end,:);
        disp('WDBC');
    end
    %%
    if (index==10)
        load('Data/echocardiogram_data.mat');
        load('Data/echocardiogram_label.mat');
        X=x;
        Y= y;
        Y(find(Y==0))=-1;
        Ctrain=X(1:80,:);
        dtrain= Y(1:80,:);
        Ctest= X(81:end,:);
        dtest= Y(81:end,:);
        disp('Echo');
    end
    
    %%
    if (index==11)
        
        load('Data/german.txt');
        X=german(:,1:end-1);
        Y= german(:,end);
        Y(find(Y==2))=-1;
        Ctrain=X(1:500,:);
        dtrain= Y(1:500,:);
        Ctest= X(501:end,:);
        dtest= Y(501:end,:);
        disp('Germans');
        
    end
    %%
    
    if (index==12)
        
        data=xlsread('Data/Australian.xlsx');
        X=data(:,2:end);
        Y= data(:,1);
        Y(find(Y==2))=-1;
        Ctrain=X(1:400,:);
        dtrain= Y(1:400,:);
        Ctest= X(401:end,:);
        dtest= Y(401:end,:);
        disp('Australian');
        
    end
    %%
    if (index==13)
        
        data=xlsread('Data/Bupa-Liver.xlsx');
        X=data(:,2:end);
        Y= data(:,1);
        Y(find(Y==0))=-1;
        Ctrain=X(1:250,:);
        dtrain= Y(1:250,:);
        Ctest= X(251:end,:);
        dtest= Y(251:end,:);
        disp('Bupa-Liver');
        
    end
    
    %%
    if (index==14)
        load('Data/votes.mat')
        X=votes(:,2:end);
        Y= votes(:,1);
        Y(find(Y==2))=-1;
        Ctrain=X(1:200,:);
        dtrain= Y(1:200,:);
        Ctest= X(201:end,:);
        dtest= Y(201:end,:);
        disp('Votes');
        
    end
    %%
    if (index==15)
        load('Data/diabetes_data.mat')
        load('Data/diabetes_label.mat')
        X= data1;
        Y= label;
        Y(find(Y==0))=-1;
        Ctrain=X(1:500,:);
        dtrain= Y(1:500,:);
        Ctest= X(501:end,:);
        dtest= Y(501:end,:);
        disp('Daibetes');
        
    end
    
    %%
    if(index==16)
        data=xlsread('Data/fertility_Diagnosis.xlsx');
        X= data(:,1:end-1);
        Y= data(:,end);
        Y(find(Y==0))=-1;
        Ctrain=X(1:50,:);
        dtrain= Y(1:50,:);
        Ctest= X(51:end,:);
        dtest= Y(51:end,:);
        disp('Fertility');
        
    end
    %%
    if(index==17)
        data= xlsread('Data/Sonar.xlsx');
        %         rng(2);
        X= data(:,2:end);
        Y= data(:,1);
        Y(find(Y==0))=-1;
        r1=randperm(size(X,1));
        X = X(r1,:);
        Y=Y(r1,:);
        Ctrain=X(1:100,:);
        dtrain= Y(1:100,:);
        Ctest= X(101:end,:);
        dtest= Y(101:end,:);
        disp('Sonar');
        
    end
    
    %%
    % %  rng(9);
    %  r1=randperm(size(Ctrain,1));
    %  Ctrain= Ctrain(r1,:);
    %  dtrain=dtrain(r1,:);
    %%
    if(index==18)
        load('Data/ecoli_data.mat');
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
        disp('Ecoil');
        
    end
    %%
    if(index==19)
        load('Data/plrx.txt');
        X=plrx(:,1:end-1);
        Y= plrx(:,end);
        Y(find(Y==2))=-1;
        Ctrain=X(1:100,:);
        dtrain= Y(1:100,:);
        Ctest= X(101:end,:);
        dtest= Y(101:end,:);
        disp('Prlx');
        
    end
    
    %%
    if (index==20)
        load('Data/monks_1_train.txt');
        load('Data/monks_1_test.txt');
        Ctrain= monks_1_train(:,2:end);
        dtrain=  monks_1_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= monks_1_test(:,2:end);
        dtest=  monks_1_test(:,1);
        dtest(find(dtest==0))=-1;
        disp('Monk 1');
    end
    if(index==21)
        load('Data/spambase_data.mat');
        X=spambase(:,1:end-1);
        Y= spambase(:,end);
        Y(find(Y==0))=-1;
        rng(1);
        r=randperm(length(Y));
        X= X(r,:);
        Y=Y(r,:);
        Ctrain=X(1:3000,:);
        dtrain= Y(1:3000,:);
        Ctest= X(3001:end,:);
        dtest= Y(3001:end,:);
        disp('Spambase');  
    end
    
    if(index==22)
        load('Data/Wine.mat');
        X=Wine(:,1:end-1);
        Y= Wine(:,end);
        Ctrain=X(1:90,:);
        dtrain= Y(1:90,:);
        Ctest= X(91:end,:);
        dtest= Y(91:end,:);
        disp('Wine');
        
    end
    if(index==23)
        load('Data/Breast_cancer(Original).mat');
        %         rng(2);
        X= A(:,1:end-1);
        Y= A(:,end);
        Ctrain=X(1:400,:);
        dtrain= Y(1:400,:);
        Ctest= X(401:end,:);
        dtest= Y(401:end,:);
        disp('Original');
        
    end
    if(index==24)
        load('Data/BUPA.mat');
        X=BUPA(:,1:end-1);
        Y= BUPA(:,end);
        Ctrain=X(1:170,:);
        dtrain= Y(1:170,:);
        Ctest= X(171:end,:);
        dtest= Y(171:end,:);
        disp('BUPA');
        
    end
    if(index==25)
        Patients = importdata('Data/Patients.mat');
        X=Patients(:,1:end-1);
        Y= Patients(:,end);
        Ctrain=X(1:800,:);
        dtrain= Y(1:800,:);
        Ctest= X(801:end,:);
        dtest= Y(801:end,:);
        disp('Patients');
        
    end
    if(index==26)
        aps_failure_test_set = importdata('aps_failure_test_set.mat');
        X = aps_failure_test_set(:,1:end-1);
        Y = aps_failure_test_set(:,end);
        Ctrain=X(1:900,:);
        dtrain= Y(1:900,:);
        Ctest= X(901:end,:);
        dtest= Y(901:end,:);
        disp('aps_failure_test_set');
        
    end
    if(index==27)
        creditcard = importdata('Data/creditcard.mat');
        X = creditcard(:,1:end-1);
        Y = creditcard(:,end);
        Ctrain=X(1:700,:);
        dtrain= Y(1:700,:);
        Ctest= X(701:end,:);
        dtest= Y(701:end,:);
        disp('creditcard');
        
    end
    if(index==28)
        creditcard = importdata('Data/AAA_data.mat');
        X = creditcard(:,1:end-1);
        Y = creditcard(:,end);
        Ctrain=X(1:90,:);
        dtrain= Y(1:90,:);
        Ctest= X(91:end,:);
        dtest= Y(91:end,:);
        disp('AAA_data');
        
    end
    
    % 加载数据集
    Data_feature = [Ctrain;Ctest]; 
    Data_label  = [dtrain;dtest];
    
    %%tsne可视化
    [reduced_data] = t_sne(Data_feature,Data_label,index);
    data_i_want = [reduced_data Data_label];
    save('Output/wine_reduced.mat', 'data_i_want');


%    %% 计算不平衡度
%    A = sum(Data_label == -1);
%    B = sum(Data_label == 1);
%     imbalance_ratio = sum(Data_label == -1) / sum(Data_label == 1);
%     if(imbalance_ratio>1)
%         imbalance_ratio = 1/imbalance_ratio;
%     end
%     fprintf('Imbalance ratio for Dataset %d: %.2f\n', index, imbalance_ratio);


%     %%异常值检测
%     [~] = noise_detec2(Data_feature,Data_label)

% %%  分布差异检验
%     [~] = ks_test(Data_feature,Data_label)


end

%%精度与数据集大小图

% % 数据
% sampleCount = [1631, 306, 690, 345, 559, 569, 341, 270, 556, 601, 554, 435, 1565, 1707, 178, 327, 768, 768, 150];
% featureCount = [78, 3, 14, 6, 32, 29, 6, 13, 6, 6, 6, 16, 31, 169, 13, 7, 8, 8, 9];
% accuracy = [74.25, 73.08, 86.55, 70.53, 98.75, 98.82, 75.86, 85.00, 68.06, 67.36, 84.03, 95.74, 96.56, 98.35, 73.86, 88.24, 79.91, 81.72, 94.00];
% 
% % 创建3D曲面图
% figure;
% scatter3(sampleCount, featureCount, accuracy, 'filled');
% xlabel('Sample Count');
% ylabel('Feature Count');
% zlabel('Accuracy');
% title('Accuracy vs. Sample Count and Feature Count (3D Scatter Plot)');
