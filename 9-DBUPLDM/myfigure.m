%p=0.125;
dbldm = [67.36111111,84.49074074,77.35849057,85.83333333,92.05298013,76.70940171,98.81656805,90.19607843,76.4,86.20689655,94.89361702,81.34328358,94,74.07407407,94.48818898,69.51219512,67.59259259];
ldm = [67.12962963,84.02777778,77.35849057,85,93.37748344,79.91452991,98.81656805,88.23529412,68,86.55172414,95.74468085,81.71641791,94,74.07407407,94.48818898,67.07317073,67.59259259];
dbupldm = [67.82407407,88.88888889,79.24528302,87.5,92.71523179,78.84615385,98.81656805,92.15686275,76.4,87.24137931,95.31914894,82.08955224,96,76.85185185,94.48818898,69.51219512,68.28703704];
upldm = [67.59259259,88.88888889,77.35849057,85.83333333,93.37748344,79.91452991,98.81656805,90.19607843,72,86.55172414,95.74468085,82.8358209,94,75.92592593,94.48818898,68.29268293,68.28703704];
dbupsvm = [67.12962963,84.72222222,78.30188679,85.83333333,92.05298013,72.64957265,95.85798817,82.35294118,67.4,86.89655172,93.19148936,72.76119403,94,71.2962963,87.4015748,70.73170732,66.43518519];
psvm = [67.12962963,84.25925926,78.30188679,86.66666667,94.0397351,67.94871795,95.85798817,76.47058824,67.2,84.48275862,94.04255319,72.76119403,94,71.2962963,94.48818898,68.29268293,65.27777778];
upsvm = [67.12962963,86.34259259,76.41509434,86.66666667,94.0397351,71.15384615,95.85798817,90.19607843,67.2,87.24137931,91.4893617,72.76119403,94,74.07407407,94.48818898,67.07317073,66.43518519];
svm = [67.12962963,81.01851852,76.41509434,86.66666667,94.0397351,67.09401709,79.28994083,70.58823529,32.8,84.48275862,85.10638298,67.91044776,94,69.44444444,86.61417323,67.07317073,62.73148148];

data=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17];
%% 
a1=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=1','B3:B19'); b1 = mean(a1);
a2=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=1','H3:H19'); b2 = mean(a2);
a3=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=1','N3:N19'); b3 = mean(a3);
a4=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=1','T3:T19'); b4 = mean(a4);
a5=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=1','B23:B39'); b5 = mean(a5);
a6=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=1','H23:H39'); b6 = mean(a6);
a7=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=1','N23:N39'); b7 = mean(a7);
a8=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=1','T23:T39'); b8 = mean(a8);

a11=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.5','B3:B19'); b11 = mean(a11);
a22=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.5','H3:H19'); b22 = mean(a22);
a33=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.5','N3:N19'); b33 = mean(a33);
a44=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.5','T3:T19'); b44 = mean(a44);
a55=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.5','B23:B39'); b55 = mean(a55);
a66=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.5','H23:H39'); b66 = mean(a66);
a77=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.5','N23:N39'); b77 = mean(a77);
a88=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.5','T23:T39'); b88 = mean(a88);

a111=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.25','B3:B19'); b111 = mean(a111);
a222=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.25','H3:H19'); b222 = mean(a222);
a333=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.25','N3:N19'); b333 = mean(a333);
a444=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.25','T3:T19'); b444 = mean(a444);
a555=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.25','B23:B39'); b555 = mean(a555);
a666=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.25','H23:H39'); b666 = mean(a666);
a777=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.25','N23:N39'); b777 = mean(a777);
a888=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.25','T23:T39'); b888 = mean(a888);

a1111=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.125','B3:B19'); b1111 = mean(a1111);
a2222=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.125','H3:H19'); b2222 = mean(a2222);
a3333=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.125','N3:N19'); b3333 = mean(a3333);
a4444=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.125','T3:T19'); b4444 = mean(a4444);
a5555=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.125','B23:B39'); b5555 = mean(a5555);
a6666=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.125','H23:H39'); b6666 = mean(a6666);
a7777=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.125','N23:N39'); b7777 = mean(a7777);
a8888=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.125','T23:T39'); b8888 = mean(a8888);

a11111=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.0625','B3:B19'); b11111 = mean(a11111);
a22222=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.0625','H3:H19'); b22222 = mean(a22222);
a33333=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.0625','N3:N19'); b33333 = mean(a33333);
a44444=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.0625','T3:T19'); b44444 = mean(a44444);
a55555=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.0625','B23:B39'); b55555 = mean(a55555);
a66666=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.0625','H23:H39'); b66666 = mean(a66666);
a77777=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.0625','N23:N39'); b77777 = mean(a77777);
a88888=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.0625','T23:T39'); b88888 = mean(a88888);

a111111=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.03125','B3:B19'); b111111 = mean(a111111);
a222222=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.03125','H3:H19'); b222222 = mean(a222222);
a333333=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.03125','N3:N19'); b333333 = mean(a333333);
a444444=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.03125','T3:T19'); b444444 = mean(a444444);
a555555=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.03125','B23:B39'); b555555 = mean(a555555);
a666666=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.03125','H23:H39'); b666666 = mean(a666666);
a777777=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.03125','N23:N39'); b777777 = mean(a777777);
a888888=xlsread('C:\Users\DDH\Desktop\DBUPLDM_data1.xlsx','p=0.03125','T23:T39'); b888888 = mean(a888888);
p=[1 1/2 1/4 1/8 1/16 1/32];

%% fig1
subplot(2,3,1);plot(data,dbupldm);hold on; plot(data,svm);legend('DBUPLDM','SVM');xlabel('Dataset');ylabel('Acuurancy');...
subplot(2,3,2);plot(data,dbupldm);hold on; plot(data,psvm);legend('DBUPLDM','PinSVM');xlabel('Dataset');ylabel('Acuurancy');...
subplot(2,3,3);plot(data,dbupldm);hold on; plot(data,upsvm);legend('DBUPLDM','UPSVM');xlabel('Dataset');ylabel('Acuurancy');...
subplot(2,3,4);plot(data,dbupldm);hold on; plot(data,dbupsvm);legend('DBUPLDM','DBUPSVM');xlabel('Dataset');ylabel('Acuurancy');...
subplot(2,3,5);plot(data,dbupldm);hold on; plot(data,ldm);legend('DBUPLDM','LDM');xlabel('Dataset');ylabel('Acuurancy');...
subplot(2,3,6);plot(data,dbupldm);hold on; plot(data,dbldm);legend('DBUPLDM','DBLDM');xlabel('Dataset');ylabel('Acuurancy'); hold off;

%% fig2
plot(p,[b6,b66,b666,b6666,b66666,b666666]);hold on;plot(p,[b4,b44,b444,b4444,b44444,b444444]);...
plot(p,[b8,b88,b888,b8888,b88888,b888888]);plot(p,[b7,b77,b777,b7777,b77777,b777777]);legend('DBUPLDM','DBUPSVM','DBLDM','LDM');


%% fig3
subplot(2,3,1);plot(data,svm);hold on; plot(data,psvm);legend('SVM','PinSVM');xlabel('Dataset');ylabel('Acuuracy');...
subplot(2,3,2);plot(data,svm);hold on; plot(data,upsvm);legend('SVM','UPSVM');xlabel('Dataset');ylabel('Acuuracy');...
subplot(2,3,3);plot(data,svm);hold on; plot(data,dbupsvm);legend('SVM','DBUPSVM');xlabel('Dataset');ylabel('Acuuracy');...
subplot(2,3,4);plot(data,svm);hold on; plot(data,ldm);legend('SVM','LDM');xlabel('Dataset');ylabel('Acuuracy');...
subplot(2,3,5);plot(data,svm);hold on; plot(data,dbldm);legend('SVM','DBLDM');xlabel('Dataset');ylabel('Acuuracy');...
subplot(2,3,6);plot(data,svm);hold on; plot(data,dbupldm);legend('SVM','DBUPLDM');xlabel('Dataset');ylabel('Acuuracy'); hold off;
