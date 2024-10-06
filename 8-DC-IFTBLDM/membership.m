function [data,y,s]=membership(trainset,Y)   %若X的列数与Y的行数不等也可以运行，自动忽略了X的其他列数


%load liang data y
%Y=y;
%trainset=data;
group=max(Y);  %矩阵来源于group组

   index=find(Y==1);
    index1=find(Y==-1);  
    A=trainset(:,index);
    B=trainset(:,index1);
    m1=mean(A');
    m2=mean(B');
    n1=size(A,2);
    n2=size(B,2);
    for i=1:n1
        d1(1,i)=norm(A(:,i)-m1);
        dd1(1,i)=norm(A(:,i)-m2);
    end
         for i=1:n2
        d2(1,i)=norm(B(:,i)-m1);
        dd2(1,i)=norm(B(:,i)-m2);
         end
   r1=max(d1);
   rr1=max(dd1);
   r2=max(d2);
   rr2=max(dd2);
    for i=1:n1
        Mem1(1,i)=1-d1(1,i)/(r1+10e-7);
        Nmem1(1,i)=1-dd1(1,i)/(rr1+10e-7);
    end
          for i=1:n2
        Mem2(1,i)=1-d2(1,i)/(r2+10e-7);
        Nmem2(1,i)=1-dd2(1,i)/(rr2+10e-7);
          end
    for i=1:n1
        s1(1,i)=sqrt((Mem1(i)^2+(1-Nmem1(1,i))^2)/2);
    end
     for i=1:n2
        s2(1,i)=sqrt((Mem2(i)^2+(1-Nmem2(1,i))^2)/2);
    end
    s=[s1 s2];
    data=[A B];
    y=[ones(n1,1); 2*ones(n2,1)];
    %s=ones(1,size(y,1));