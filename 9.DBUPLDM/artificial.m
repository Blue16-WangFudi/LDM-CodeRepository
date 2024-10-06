%%
close all;
clear all;
%% 
clc
rand('state', 2015)
randn('state', 2015)
i=0;
%% 
for ii = 0:5:10
    i=i+1;
    j=0;
    for jj = 0:5:10
        fprintf(' \n[ii=%1.0f,jj=%1.0f] \n',ii,jj);
        j=j+1;
        % 第一组
        mul=[0.5,-3.5]; % 均值
        S1=[0.2 0;0 3]; % 协方差
        data1=mvnrnd(mul, S1, 50); % 产生高斯分布数据
        data1(:,3)=1;
        % 第二组数据
        mu2=[-0.5,3.5];
        S2=[0.2 0;0 3];
        data2=mvnrnd(mu2,S2,250);
        data2(:,3)=-1;
        % noises of p
        mm1=ii;
        mu3=[0,0];
        S3=[1 -0.8;-0.8 1];
        data3=mvnrnd(mu3,S3,mm1);
        data3(:,3)=-1;
        % noises of n
        mm2=jj;
        mu4=[0,0];
        S4=[1 -0.8;-0.8 1];
        data4=mvnrnd(mu4,S4,mm2);
        data4(:,3)=1;
        data_noise = [data3;data4];
        % predict
        mu5=[0.5,-2.8]; % 均值
        S5=[0.2 0;0 2]; % 协方差
        data5=mvnrnd(mu5, S5, 50); % 产生高斯分布数据
        data5(:,3)=1;
        % 第二组数据
        mu55=[-0.5,3];
        S55=[0.2 0;0 2];
        data55=mvnrnd(mu55,S55,50);
        data55(:,3)=-1;
        %% all
        data_train=[data1;data2;data_noise];
        data_predict=[data5;data55];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
        %% Parameter setting
        Ctrain= svdatanorm(Ctrain,'svpline');
        Ctest= svdatanorm(Ctest,'scpline');
        
        %s = Fuzzy_MemberShip(Ctrain, dtrain, kernel, 0.5,p1);
        clear C;
        kernel=1;
        C0= 2^0;
        tau=0;
        svm_tau=0;
        p1= 2^-2;
        lamb1 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0];
        lamb2 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0];
        p1val=[2^-6];
        c1val=[2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3];
        theta = Balance_factor(Ctrain, dtrain, kernel, 0.5,p1);
        %c1val=[1 10 50 100];
        %     p1val=2;
        %     c1val=2^-2;
        %% Prameter Tunning
        %     [acc_svm, opt_p1,opt_c1,t1]= tune_para_svm(Ctrain,dtrain,Ctest,dtest,kernel,c1val,p1val);
        %     p1= opt_p1;
        %     pp1(index,1)= p1;
        %     C0= opt_c1;
        %     lamb1 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
        %     lamb2 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
        %       lamb1 = 0;
        %       lamb2 = 0;
        %     if (kernel==2)
        %         fprintf('\n Optimal Accuracy = %3.2f with kernel parameter p1= %3.4f and C= %3.4f',acc_svm, opt_p1,opt_c1);
        %     end
        %     if (kernel==1)
        %         fprintf('\n Optimal Accuracy = %3.2f with  C= %3.4f',acc_svm,opt_c1);
        %     end
        %     fprintf('\n Time Elpased  in Tunning Paramter = %3.2f seconds',t1);
        
        %%
        [opt_tau1,opt_tau2,opt_tau3,opt_tau4,opt_tau5,...
            acc_svm,acc_upsvm,acc_psvm,acc_csupsvm,acc_upldm,acc_csupldm,acc_ldm,acc_csldm,...
            time0,time1,time2,time3,time4,time5,time6,time7,...
            opt_C0,opt_C1,opt_C2,opt_C3,opt_C4,opt_C5,opt_C6,opt_C7,h]= tune_tau(Ctrain,dtrain,Ctest,dtest,c1val,kernel,p1,theta,lamb1,lamb2);
        %% SVM
        fprintf('\n SVM Accuracy=%3.2f,time = %3.2f,C = %3.2f',acc_svm,time0,opt_C0);
        d0= [acc_svm,time0,p1,opt_C0,0];
        
        %% UPSVM
        fprintf('\n UPSVM Accuracy=%3.2f,time = %3.2f,tau = %3.2f,C = %3.2f',acc_upsvm,time1,opt_tau1,opt_C1);
        d1= [acc_upsvm,time1,p1,opt_C1,opt_tau1];
        
        %% PSVM
        fprintf('\n PSVM Accuracy=%3.2f,time = %3.2f,tau = %3.2f,C = %3.2f',acc_psvm,time2,opt_tau2,opt_C2);
        d2= [acc_psvm,time2,p1,opt_C2,opt_tau2];
        
        
        %% FUPSVM
        fprintf('\n DBUPSVM Accuracy=%3.2f,time = %3.2f,tau = %3.2f,C = %3.2f',acc_csupsvm,time3,opt_tau3,opt_C3);
        d3= [acc_csupsvm,time3,p1,opt_C3,opt_tau3];
        
        %% UPLDM
        fprintf('\n UPLDM Accuracy=%3.2f,time = %3.2f,tau = %3.2f,C = %3.2f',acc_upldm,time4,opt_tau4,opt_C4);
        d4= [acc_upldm,time4,p1,opt_C4,opt_tau4];
        
        %% FUPLDM
        fprintf('\n DBUPLDM Accuracy=%3.2f,time = %3.2f,tau = %3.2f,C = %3.2f',acc_csupldm,time5,opt_tau5,opt_C5);
        d5= [acc_csupldm,time5,p1,opt_C5,opt_tau5];
        
        %% LDM
        fprintf('\n LDM Accuracy=%3.2f,time = %3.2f,C = %3.2f',acc_ldm,time6,opt_C6);
        d6= [acc_ldm,time6,p1,opt_C6,0];
        
        %% F-LDM
        fprintf('\n DBLDM Accuracy=%3.2f,time = %3.2f,C = %3.2f',acc_csldm,time7,opt_C7);
        d7= [acc_csldm,time7,p1,opt_C7,0];
        %%
        acc1(i,j) = acc_svm;acc2(i,j) = acc_upsvm;acc3(i,j) = acc_psvm;acc4(i,j) = acc_csupsvm;
        acc5(i,j) = acc_ldm;acc6(i,j) = acc_csldm;acc7(i,j) = acc_upldm;acc8(i,j) = acc_csupldm;
    end
end
% dd2(index,:)= [d0,d1,d2,d3,d4,d5,d6,d7];
% Ymat=[acc1,acc2,acc3,acc4,acc5,acc6,acc7];
% h=createfigure(0:10:80,Ymat);
ACC = [acc1,acc2,acc3,acc4,acc5,acc6,acc7,acc8];
plot([0 5 10],[acc1(1,1),acc1(2,2),acc1(3,3)]);
hold on
plot([0 5 10],[acc3(1,1),acc3(2,2),acc3(3,3)]);
plot([0 5 10],[acc2(1,1),acc2(2,2),acc2(3,3)]);
plot([0 5 10],[acc5(1,1),acc5(2,2),acc5(3,3)]);
plot([0 5 10],[acc8(1,1),acc8(2,2),acc8(3,3)]);
legend('SVM','PinSVM','UPSVM','LDM','DBUPLDM');
hold off
figure
surf([0 5 10],[0 5 10],acc1);
xlabel('m_p'),ylabel('m_n'),zlabel('Acc');
title('SVM');
shading interp;
zlim([90 100])
colormap(jet);
figure
surf([0 5 10],[0 5 10],acc3);
xlabel('m_p'),ylabel('m_n'),zlabel('Acc');
title('PinSVM');
shading interp;
zlim([90 100])
colormap(jet);
figure
surf([0 5 10],[0 5 10],acc2);
xlabel('m_p'),ylabel('m_n'),zlabel('Acc');
title('UPSVM');
shading interp;
zlim([90 100])
colormap(jet);
% figure
% surf([0 5 10 15 20 25 30],[0 5 10 15 20 25 30],acc6);
% xlabel('m_p'),ylabel('m_n'),zlabel('Acc');
% title('F-LDM');
% shading interp;
% zlim([90 100])
% colormap(jet);
% figure
% surf([0 5 10 15 20 25 30],[0 5 10 15 20 25 30],acc4);
% xlabel('m_p'),ylabel('m_n'),zlabel('Acc');
% title('F-UPSVM');
% shading interp;
% colormap(gray);
figure
surf([0 5 10],[0 5 10],acc5);
xlabel('m_p'),ylabel('m_n'),zlabel('Acc');
title('LDM');
shading interp;
zlim([90 100])
colormap(jet);
% figure
% surf([0 5 10 15 20 25 30],[0 5 10 15 20 25 30],acc6);
% xlabel('m_p'),ylabel('m_n'),zlabel('Acc');
% title('F-LDM');
% shading interp;
% zlim([90 100])
% colormap(jet);
% figure
% surf([0 5 10],[0 5 10],acc7);
% xlabel('m_p'),ylabel('m_n'),zlabel('Acc');
% title('UPLDM');
% shading interp;
% colormap(jet);
figure
surf([0 5 10],[0 5 10],acc8);
xlabel('m_p'),ylabel('m_n'),zlabel('Acc');
title('DBUPLDM');
shading interp;
zlim([90 100])
colormap(jet);

% 
plot([0 10 20],[acc1(1,1),acc1(2,2),acc1(3,3)]);
hold on
plot([0 10 20],[acc3(1,1),acc3(2,2),acc3(3,3)]);
plot([0 10 20],[acc2(1,1),acc2(2,2),acc2(3,3)]);
plot([0 10 20],[acc5(1,1),acc5(2,2),acc5(3,3)]);
plot([0 10 20],[acc8(1,1),acc8(2,2),acc8(3,3)]);
legend('SVM','PinSVM','UPSVM','LDM','DBUPLDM');
hold off