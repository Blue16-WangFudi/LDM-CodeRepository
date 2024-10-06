function [opt_tau1,opt_tau2,opt_tau3,opt_tau4,opt_tau5,...
    acc_svm,acc_upsvm,acc_psvm,acc_fupsvm,acc_upldm,acc_fupldm,acc_ldm,acc_fldm,...
    time0,time1,time2,time3,time4,time5,time6,time7,...
    opt_C0,opt_C1,opt_C2,opt_C3,opt_C4,opt_C5,opt_C6,opt_C7,h]= tune_tau(Ctrain,dtrain,Ctest,dtest,C,kernel,p1,s,lamb1,lamb2)
tauval= 1;
acc_k4 = zeros(length(lamb1),length(lamb2));acc_k5 = zeros(length(lamb1),length(lamb2));
acc_k6 = zeros(length(lamb1),length(lamb2));acc_k7 = zeros(length(lamb1),length(lamb2));
C_k4 = zeros(length(lamb1),length(lamb2));C_k5 = zeros(length(lamb1),length(lamb2));
C_k6 = zeros(length(lamb1),length(lamb2));C_k7 = zeros(length(lamb1),length(lamb2));
acc0_ = zeros(length(tauval),1);acc1_ = zeros(length(tauval),1);acc2_ = zeros(length(tauval),1);
acc3_ = zeros(length(tauval),1);acc4_ = zeros(length(tauval),1);acc5_ = zeros(length(tauval),1);
acc6_ = zeros(length(tauval),1);acc7_ = zeros(length(tauval),1);
% std0 = zeros(length(tauval),1);std1 = zeros(length(tauval),1);std2 = zeros(length(tauval),1);
% std3 = zeros(length(tauval),1);std4 = zeros(length(tauval),1);std5 = zeros(length(tauval),1);
% std6 = zeros(length(tauval),1);std7 = zeros(length(tauval),1);
count=0;
%%
for jj=1:length(tauval)
%     for t = 1:10
        time0=0;time1=0;time2=0;time3=0;time4=0;time5=0;time6=0;time7=0;
        fprintf('%3.0f steps remaining...\n',10*length(tauval)-count);
        tau= tauval(jj);
        tic
        [acc00(t),C0] = Unified_pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, 0,C,p1);  %SVM
        time0 = time0 + toc;
        tic
        [acc11(t),C1] = Unified_pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1);  %UPSVM
        time1 = time1 + toc;
        tic
        [acc22(t),C2] = pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1); %PSVM
        time2 = time2 + toc;
        tic
        [acc33(t),C3] = Unified_pin_fsvm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1,s); %FUPSVM
        time3 = time3 + toc;
        for k1 = 1:length(lamb1)
            for k2 = 1:length(lamb2)
                tic
                [acc44(t),C4] = Unified_pin_ldm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1,lamb1(k1),lamb2(k2));  %UPLDM
                time4 = time4 + toc;
                tic
                [acc55(t),C5] = Unified_pin_fldm(Ctrain, dtrain, Ctest,dtest, kernel, tau,C,p1,s,lamb1(k1),lamb2(k2));  %FUPLDM
                time5 = time5 + toc;
                tic
                [acc66(t),C6] = Unified_pin_ldm(Ctrain, dtrain, Ctest,dtest, kernel, 0,C,p1,lamb1(k1),lamb2(k2));  %LDM
                time6 = time6 + toc;
                tic
                [acc77(t),C7] = Unified_pin_fldm(Ctrain, dtrain, Ctest,dtest, kernel, 0,C,p1,s,lamb1(k1),lamb2(k2));  %FLDM
                time7 = time7 + toc;
            end
            acc_k4(k1,k2)= mean(acc44);
            acc_k5(k1,k2)= mean(acc55);
            acc_k6(k1,k2)= mean(acc66);
            acc_k7(k1,k2)= mean(acc77);
            %std_k4(k1,k2) = std(acc44);std_k5(k1,k2) = std(acc55);std_k6(k1,k2) = std(acc66);std_k7(k1,k2) = std(acc77);
            C_k4(k1,k2)= C4;
            C_k5(k1,k2)= C5;
            C_k6(k1,k2)= C6;
            C_k7(k1,k2)= C7;
        end
        count=count+1;
%     end
    
    acc0_(jj)= mean(acc00);
    acc1_(jj)= mean(acc11);
    acc2_(jj)= mean(acc22);
    acc3_(jj)= mean(acc33);
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
    C4_(jj)= C_k4(xx4,yy4);
    C5_(jj)= C_k5(xx5,yy5);
    C6_(jj)= C_k6(xx6,yy6);
    C7_(jj)= C_k7(xx7,yy7);
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
    
end
Ymat=[acc1_',acc2_',acc5_'];
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
[acc_fupsvm,i3]=max(acc2_);
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
opt_C5 = C5_(i5);
opt_C6 = C6_(i6);
opt_C7 = C7_(i7);
end