function U = Fuzzy_MemberShip_FCM(data, label)
%--------------------------------
cluster_n = 2;
[center,S ,obj_fcn] = fcm(data , 2); % æ€¿‡÷––ƒ
data_p = data(label==1,:);
data_n = data(label==-1,:);
for i=1:length(data_p)
    U(i)=0.9*S(1,i)+0.1*S(2,i); %class 1
end

for i=length(data_p)+1:length(data)
    U(i)=0.9*S(2,i)+0.1*S(1,i); %class 2
end
U=U';
end