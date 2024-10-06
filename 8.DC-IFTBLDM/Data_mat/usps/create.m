% create data
clear;
load('usps.mat');
TFS_train = zeros(500,257);
TFS_test = zeros(750,257);
n=0;
m=0;
for i = 0:9
    if mod(i,2) == 0
        n=n+1;
        usps_n = usps(find(usps(:,end)==i),:);        
        ab = usps_n(randperm(size(usps_n,1),200),:);
        TFS_train((1+100*(n-1)):(100*n),:)=ab(1:100,:);       
        TFS_test((1+100*(n-1)):(100*n),:)=ab(101:200,:); 
    end
    if mod(i,2) == 1
        m=m+1;
        usps_m = usps(find(usps(:,end)==i),:);        
        ab_ = usps_m(randperm(size(usps_m,1),50),:);
        TFS_test((501+50*(m-1)):(500+50*m),:)=ab_;        
    end
       
end
