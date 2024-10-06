function [opt_tau1,opt_tau2,opt_tau3,opt_tau4,opt_tau5,...
    acc_svm,acc_upsvm,acc_psvm,acc_fupsvm,acc_upldm,acc_fupldm,acc_ldm,acc_fldm,...
    time0,time1,time2,time3,time4,time5,time6,time7,...
    opt_C0,opt_C1,opt_C2,opt_C3,opt_C4,opt_C5,opt_C6,opt_C7,h,opt_mean0,opt_variance0,opt_mean1,opt_variance1,opt_mean2,opt_variance2,opt_mean4,opt_variance4,opt_mean5,opt_variance5,opt_mean6,opt_variance6,opt_mean7,opt_variance7]= tune_tau(Ctrain,dtrain,Ctest,dtest,C,kernel,p1,s,lamb1,lamb2)
tauval= -1:0.1:0;
acc_k4 = zeros(length(lamb1),length(lamb2));acc_k5 = zeros(length(lamb1),length(lamb2));
acc_k6 = zeros(length(lamb1),length(lamb2));acc_k7 = zeros(length(lamb1),length(lamb2));
C_k4 = zeros(length(lamb1),length(lamb2));C_k5 = zeros(length(lamb1),length(lamb2));
C_k6 = zeros(length(lamb1),length(lamb2));C_k7 = zeros(length(lamb1),length(lamb2));
acc0_ = zeros(1,length(tauval));acc1_ = zeros(1,length(tauval));acc2_ = zeros(1,length(tauval));
acc3_ = zeros(1,length(tauval));acc4_ = zeros(1,length(tauval));acc5_ = zeros(1,length(tauval));
acc6_ = zeros(1,length(tauval));acc7_ = zeros(1,length(tauval));count=0;


Margin_mean4 = zeros(length(lamb1),length(lamb2));
Margin_mean5 = zeros(length(lamb1),length(lamb2));
Margin_mean6 = zeros(length(lamb1),length(lamb2));
Margin_mean7 = zeros(length(lamb1),length(lamb2));

Margin_variance4 = zeros(length(lamb1),length(lamb2));
Margin_variance5 = zeros(length(lamb1),length(lamb2));
Margin_variance6 = zeros(length(lamb1),length(lamb2));
Margin_variance7 = zeros(length(lamb1),length(lamb2));


%%
for jj=1:length(tauval)
        time0=0;time1=0;time2=0;time3=0;
        fprintf('%3.0f steps remaining...\n',length(tauval)-count);
        tau= tauval(jj);
        tic
        [margin_mean0,margin_variance0,acc0,C0] = Unified_pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, 0,C,p1);  %SVM
        time0 = time0 + toc;
        tic
        [margin_mean1,margin_variance1,acc1,C1] = Unified_pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1);  %UPSVM
        time1 = time1 + toc;
        tic
        [margin_mean2,margin_variance2,acc2,C2] = pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1); %PSVM
        time2 = time2 + toc;
        tic
        [acc3,C3] = Unified_pin_fsvm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1,s); %FUPSVM
        time3 = time3 + toc;
        for k1 = 1:length(lamb1)
            for k2 = 1:length(lamb2)
                time4=0;time5=0;time6=0;time7=0;
                tic
                [margin_mean4,margin_variance4,acc4,C4] = Unified_pin_ldm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1,lamb1(k1),lamb2(k2));  %UPLDM
                time4 = time4 + toc;
                tic
                [margin_mean5,margin_variance5,acc5,C5] = Unified_pin_fldm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1,s,lamb1(k1),lamb2(k2));  %FUPLDM
                time5 = time5 + toc;
                tic
                [margin_mean6,margin_variance6,acc6,C6] = Unified_pin_ldm(Ctrain, dtrain, Ctest,dtest, kernel, 0,C,p1,lamb1(k1),lamb2(k2));  %LDM
                time6 = time6 + toc;
                tic
                [margin_mean7,margin_variance7,acc7,C7] = Unified_pin_fldm(Ctrain, dtrain, Ctest,dtest, kernel, 0,C,p1,s,lamb1(k1),lamb2(k2));  %FLDM
                time7 = time7 + toc;
            end
            acc_k4(k1,k2)= acc4;
            acc_k5(k1,k2)= acc5;
            acc_k6(k1,k2)= acc6;
            acc_k7(k1,k2)= acc7;
            C_k4(k1,k2)= C4;
            C_k5(k1,k2)= C5;
            C_k6(k1,k2)= C6;
            C_k7(k1,k2)= C7;
            
            Margin_mean4 (k1,k2) = margin_mean4;
            Margin_mean5 (k1,k2) = margin_mean5;
            Margin_mean6 (k1,k2) = margin_mean6;
            Margin_mean7 (k1,k2) = margin_mean7;
            
            Margin_variance4 (k1,k2) = margin_variance4;
            Margin_variance5 (k1,k2) = margin_variance5;
            Margin_variance6 (k1,k2) = margin_variance6;
            Margin_variance7 (k1,k2) = margin_variance7;  
            
        end
    
    acc0_(jj)= acc0;
    acc1_(jj)= acc1;
    acc2_(jj)= acc2;
    acc3_(jj)= acc3;
    acc4_(jj)= max(max(acc_k4));
    acc5_(jj)= max(max(acc_k5));
    acc6_(jj)= max(max(acc_k6));
    acc7_(jj)= max(max(acc_k7));
    [x4,y4]=find(acc_k4==max(max(acc_k4)));xx4=x4(1);yy4=y4(1);
    [x5,y5]=find(acc_k5==max(max(acc_k5)));xx5=x5(1);yy5=y5(1);
    [x6,y6]=find(acc_k6==max(max(acc_k6)));xx6=x6(1);yy6=y6(1);
    [x7,y7]=find(acc_k7==max(max(acc_k7)));xx7=x7(1);yy7=y7(1);
    C0_(jj)= C0;
    C1_(jj)= C1;
    C2_(jj)= C2;
    C3_(jj)= C3;
    Margin_mean0_(jj) = margin_mean0;
    Margin_variance0_(jj) = margin_variance0;
    Margin_mean1_(jj) = margin_mean1;
    Margin_variance1_(jj) = margin_variance1;    
    Margin_mean2_(jj) = margin_mean2;
    Margin_variance2_(jj) = margin_variance2;    
    
    C4_(jj)= C_k4(xx4,yy4);
    C5_(jj)= C_k5(xx5,yy5);
    C6_(jj)= C_k6(xx6,yy6);
    C7_(jj)= C_k7(xx7,yy7);
    Margin_mean4_ (jj) = Margin_mean4(xx4,yy4);
    Margin_variance4_ (jj) = Margin_variance4(xx4,yy4);
    Margin_mean5_ (jj) = Margin_mean5(xx5,yy5);
    Margin_variance5_ (jj) = Margin_variance5(xx5,yy5);
    Margin_mean6_ (jj) = Margin_mean6(xx6,yy6);
    Margin_variance6_ (jj) = Margin_variance6(xx6,yy6);
    Margin_mean7_ (jj) = Margin_mean7(xx7,yy7);
    Margin_variance7_ (jj) = Margin_variance7(xx7,yy7);
    
    
%     std0(jj) = std(acc00);std1(jj) = std(acc11);std2(jj) = std(acc22);std3(jj) = std(acc33);
%     std4(jj) = std_k4(xx4,yy4);std5(jj) = std_k5(xx5,yy5);std6(jj) = std_k6(xx6,yy6);std7(jj) = std_k7(xx7,yy7);
    %     figure
    %     surf(lamb1,lamb2,acc_k3);
    %     xlabel('lambda1'),ylabel('lambda2'),zlabel('Acc');
    %     title('UPLDM');
    %     shading interp;
    %     colormap(gray);
    %     figure
    %     surf(lamb1,lamb2,acc_k4);
    %     xlabel('lambda1'),ylabel('lambda2'),zlabel('Acc');
    %     title('F-UPLDM');
    %     shading interp;
    %     colormap(gray);
    %%
    % [trainCorr1, testCorr1, cpu_time1, fold_correctness1]=n_twsvm(data,label,c, kernel, kerpara,  k, output);
    % [trainCorr2, testCorr2, cpu_time2, fold_correctness2]=mbsvm(data,label,c1, kernel, kerpara,  k, output)
    % x=fold_correctness1;
    % y=fold_correctness2;ll
    % [h,p]=ttest(x,y,0.5)
   count = count+1; 
end
Ymat=[acc0_',acc1_',acc2_',acc4_',acc6_'];
h=createfigure(tauval,Ymat);
% plot(tauval,acc,'.--'); %UPSVM
% hold on;
% plot(tauval,acc1,'.--'); %PSVM
% %plot(tauval,acc2,'.--'); %FUPSVM
% plot(tauval,acc3,'.--'); %FPLDM
% plot(tauval,acc4,'+--'); %FUPLDM
% legend('UPSVM','PSVM','FPLDM','FUPLDM');
[acc_svm,i0]=max(acc0_);
[acc_upsvm,i1]=max(acc1_);
[acc_psvm,i2]=max(acc2_);
[acc_fupsvm,i3]=max(acc3_);
[acc_upldm,i4] = max(acc4_);
[acc_fupldm,i5] = max(acc5_);
[acc_ldm,i6] = max(acc6_);
[acc_fldm,i7] = max(acc7_);
opt_tau1=tauval(i1);
opt_tau2=tauval(i2);
opt_tau3=tauval(i3);
opt_tau4=tauval(i4);
opt_tau5=tauval(i5);
opt_C0 = C0_(i0);
opt_C1 = C1_(i1);
opt_C2 = C2_(i2);
opt_C3 = C3_(i3);
opt_C4 = C4_(i4);
opt_mean0 = Margin_mean0_(i0);
opt_variance0 = Margin_variance0_(i0);
opt_mean1 = Margin_mean1_(i1);
opt_variance1 = Margin_variance1_(i1);
opt_mean2 = Margin_mean2_(i2);
opt_variance2 = Margin_variance2_(i2);

opt_mean4 = Margin_mean4_(i4);
opt_variance4 = Margin_variance4_(i4);
opt_C5 = C5_(i5);
opt_mean5 = Margin_mean5_(i5);
opt_variance5 = Margin_variance5_(i5);
opt_C6 = C6_(i6);
opt_mean6 = Margin_mean6_(i6);
opt_variance6 = Margin_variance6_(i6);
opt_C7 = C7_(i7);
opt_mean7 = Margin_mean7_(i7);
opt_variance7 = Margin_variance7_(i7);
end