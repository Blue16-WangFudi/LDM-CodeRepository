clear;
load('TFS_train.mat');
load('TFS_test.mat');
% TFS_train(find(TFS_train(:,end)~=2),end)=-1;
% TFS_train(find(TFS_train(:,end)==2),end)=1;
% 
% 
% TFS_test(find(TFS_test(:,end)~=2),end)=-1;
% TFS_test(find(TFS_test(:,end)==2),end)=1;

str{1}='train_0';
str{2}='test_0';
str{3}='train_2';
str{4}='test_2';
str{5}='train_4';
str{6}='test_4';
str{7}='train_6';
str{8}='test_6';
str{9}='train_8';
str{10}='test_8';

n=0;
for i=0:9
    load('TFS_train.mat');
    load('TFS_test.mat');
    if mod(i,2)==0
        n=n+1;
        TFS_train(find(TFS_train(:,end)~=i),end)=-1;
        TFS_train(find(TFS_train(:,end)==i),end)=1;
        eval([str{n},'=TFS_train;']);
        save(str{n},str{n}); 
        
        n=n+1;
        TFS_test(find(TFS_test(:,end)~=i),end)=-1;
        TFS_test(find(TFS_test(:,end)==i),end)=1;
        eval([str{n},'=TFS_test;']);
        save(str{n},str{n}); 
    end
end


s=sum(test_8(:,end)==-1);
% 
% 
% a = 'c';
% b = [3,4];
% eval([a,'=b;'])

