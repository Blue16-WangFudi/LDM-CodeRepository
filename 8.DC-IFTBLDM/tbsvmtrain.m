function  [ftsvm_struct] = tbsvmtrain(Traindata,Trainlable,Parameter)
% Function:  train cdftsvm
% Input:      
% Traindata         -  the train data where the feature are stored
% Trainlabel        -  the  lable of train data  
% Parameter         -  the parameters for ftsvm
%
% Output:    
% ftsvm_struct      -  ftsvm model
%
% Author: Bin-Bin Gao (csgaobb@gmail.com)
% Created on 2014.10.10
% Last modified on 2015.07.16

% check correct number of arguments
if ( nargin>3||nargin<3) 
    help  ftsvmtrain
end

ker=Parameter.ker;
CC=Parameter.CC;
CR=Parameter.CR;
Parameter.autoScale=0;
%Parameter.showplots=0;
autoScale=Parameter.autoScale;


st1 = cputime;
[groupIndex, groupString] = grp2idx(Trainlable);
groupIndex = 1 - (2* (groupIndex-1));

% %�޸�
% groupIndex = -groupIndex;
scaleData = [];

% normalization
if autoScale
    scaleData.shift = - mean(Traindata);
    stdVals = std(Traindata);
    scaleData.scaleFactor = 1./stdVals;
    % leave zero-variance data unscaled:
    scaleData.scaleFactor(~isfinite(scaleData.scaleFactor)) = 1;
    % shift and scale columns of data matrix:
    for k = 1:size(Traindata, 2)
        scTraindata(:,k) = scaleData.scaleFactor(k) * ...
            (Traindata(:,k) +  scaleData.shift(k));
    end
else
    scTraindata= Traindata;
end


Xp=scTraindata(groupIndex==1,:);
Lp=Trainlable(groupIndex==1);
Xn=scTraindata(groupIndex==-1,:);
Ln=Trainlable(groupIndex==-1);
X=[Xp;Xn];
L=[Lp;Ln];
% compute fuzzy membership
[sp,sn,NXpv,NXnv]=fuzzy(Xp,Xn,Parameter);

lp=sum(groupIndex==1);
ln=sum(groupIndex==-1);
% kernel matrix
switch ker
    case 'linear'
        kfun = @linear_kernel;kfunargs ={};
    case 'quadratic'
        kfun = @quadratic_kernel;kfunargs={};
    case 'radial'
        p1=Parameter.p1;
        kfun = @rbf_kernel;kfunargs = {p1};
    case 'rbf'
        p1=Parameter.p1;
        kfun = @rbf_kernel;kfunargs = {p1};
    case 'polynomial'
        p1=Parameter.p1;
        kfun = @poly_kernel;kfunargs = {p1};
    case 'mlp'
        p1=Parameter.p1;
        p2=Parameter.p2;
        kfun = @mlp_kernel;kfunargs = {p1, p2};
end
% kernel function
switch ker
    case 'linear'
        Kpx=Xp;Knx=Xn;
    case 'rbf'
        Kpx = feval(kfun,Xp,X,kfunargs{:});%K(X+,X)
        Knx = feval(kfun,Xn,X,kfunargs{:});%K(X-,X)
end
S=[Kpx ones(lp,1)];R=[Knx ones(ln,1)];

CC1=CC*ones(size(Xn,1),1);
CC2=CC*ones(size(Xp,1),1);

fprintf('Optimising ...\n');
switch  Parameter.algorithm
    case  'CD'
        [alpha ,vp] =  L1CD(S,R,CR,CC1);
        [beta , vn] =  L1CD(R,S,CR,CC2);
        vn=-vn;
    case  'qp'
        QR=(S'*S+CR*eye(size(S'*S)))\R';
        RQR=R*QR;
        RQR=(RQR+RQR')/2;
        
        QS=(R'*R+CR*eye(size(R'*R)))\S';
        SQS=S*QS;
        SQS=(SQS+SQS')/2;

        [alpha,~,~]=qp(RQR,-ones(ln,1),[],[],zeros(ln,1),CC1,ones(ln,1));
        [beta,~,~] =qp(SQS,-ones(lp,1),[],[],zeros(lp,1),CC2,ones(lp,1));
        
        vp=-QR*alpha;
        vn=QS*beta;
    case  'QP'
        QR=(S'*S+CR*eye(size(S'*S)))\R';
        RQR=R*QR;
        RQR=(RQR+RQR')/2;
        
        QS=(R'*R+CR*eye(size(R'*R)))\S';
        SQS=S*QS;
        SQS=(SQS+SQS')/2;
        % Solve the Optimisation Problem  
        qp_opts = optimset('display','off');
        [alpha,~,~]=quadprog(RQR,-ones(ln,1),[],[],[],[],zeros(ln,1),CC1,zeros(ln,1),qp_opts);
        [beta,~,~]=quadprog(SQS,-ones(lp,1),[],[],[],[],zeros(lp,1),CC2,zeros(lp,1),qp_opts);
        
        vp=-QR*alpha;
        vn=QS*beta;
    case   'ldm'          %%%%%%%%%%%%%%ldm算法部分
        
        lamda1=0.125;
        lamda2=0.125;
        G=R;
        H=S;
        Kx=[Kpx;Knx];
        M=[Kx,ones(lp+ln,1)];
        P=lamda1/(lp+ln)^2.*M'*((lp+ln).*eye(lp+ln)-L*L')*M+H'*H;
        Hp=(-1).*G*inv(P)*G';
        Q=lamda2/(lp+ln).*M'*L;
        f1=(ones(ln,1)'-Q'*inv(P)*G');
        qp_opts = optimset('display','off');
%         Hp=(Hp+Hp')/2;
%         f1=(f1+f1')/2;
        [alpha,~,~]=quadprog(Hp,f1,[],[],[],[],zeros(ln,1),CC1,zeros(ln,1),qp_opts);
        Sn=lamda1/(lp+ln)^2.*M'*((lp+ln).*eye(lp+ln)-L*L')*M+G'*G;
        Hn=(-1).*H*inv(Sn)*H';
        f2=(ones(lp,1)'-Q'*inv(Sn)*H');
%         Hn=(Hn+Hn')/2;
%         f2=(f2+f2')/2;
        [beta,~,~]=quadprog(Hn,f2,[],[],[],[],zeros(lp,1),CC2,zeros(lp,1),qp_opts);   
       
        QR=(S'*S+CR*eye(size(S'*S)))\R'; 
        QS=(R'*R+CR*eye(size(R'*R)))\S';
        vp=-QR*alpha;
        vn=QS*beta;
end
ExpendTime=cputime - st1;

ftsvm_struct.scaleData=scaleData;

ftsvm_struct.X = X;
ftsvm_struct.L = L;
ftsvm_struct.sp = sp;
ftsvm_struct.sn = sn;


ftsvm_struct.alpha = alpha;
ftsvm_struct.beta  = beta;
ftsvm_struct.vp = vp;
ftsvm_struct.vn = vn;

ftsvm_struct.KernelFunction = kfun;
ftsvm_struct.KernelFunctionArgs = kfunargs;
ftsvm_struct.Parameter = Parameter;
ftsvm_struct.groupString=groupString;
ftsvm_struct.time=ExpendTime;

ftsvm_struct.NXpv=NXpv;
ftsvm_struct.NXnv=NXnv;
ftsvm_struct.nv=length(NXpv)+length(NXnv);
if  Parameter.showplots
    ftsvmplot(ftsvm_struct,Traindata,Trainlable);
end   
end






