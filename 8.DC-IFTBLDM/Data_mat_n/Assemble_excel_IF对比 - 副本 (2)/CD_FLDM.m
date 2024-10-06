function  [u, beta] = CD_FLDM(Q, KY, lambda2_eORs, C_eORs, eps_bound, max_iter)

% Function:  Coordinate Descent method for the dual problem of F_LDM
%------------------- Input -------------------%
% Q, P(=(K*Y)'), lambda2_eORs, C_eORs, eps, max_iter

%------------------- Output -------------------%
% u, beta


% Author: Wendong Wang
%  Email: d.sylan@foxmail.com
%   data: 2015,9,8
% updata: 2015,10,7

% This is the first version of our CD_FLDM algorithm and it extends the work from Binbin Gao. One can find more in following webs
% 1. http://lamda.nju.edu.cn/gaobb/
% 2. http://github.com/gaobb/CDFTSVM

 rand('state', 2015)
 randn('state', 2015)
 
if  nargin>6||nargin<4   % check correct number of arguments
    help CD_FLDM
else
    if nargin<5
        eps_bound = 1e-3;
    end
    if nargin<6
        max_iter = 200;
    end

%     H_Diag = diag(KY'*(Q\KY));
%     invQKY = Q\KY;
    
    H_Diag = diag(KY'*inv(Q)*KY);
    invQKY = inv(Q)*KY;
    
    m = size(KY, 2);   
    X_new = 1:m;
    X_old = 1:m;
    
    beta  = zeros(m, 1); 
    betaold = zeros(m, 1);
%     u = Q\KY*lambda2_eORs/m; 
    u = inv(Q)*KY*lambda2_eORs/m; 
    
    PGmax_old = inf;       %M_bar
    PGmin_old = -inf;      %m_bar
    
    iter = 0;    
    while iter<max_iter
        %1 While
        PGmax_new = -inf;   %M
        PGmin_new = inf;   %m
        R = length(X_old);
        X_old = X_old(randperm(R));
    
        for  j = 1:R
            i = X_old(j);
            pg = KY(:,i)'*u-1;  
            PG = 0;               
            if beta(i) == 0
                if pg>PGmax_old
                    X_new(X_new == i) = [];
                    continue;
                elseif  pg<0
                    PG = pg;
                end
            elseif beta(i) == C_eORs(i)/m
                if pg<PGmin_old
                    X_new(X_new == i) = [];
                    continue;
                elseif  pg>0
                    PG = pg;
                end
            else
                PG = pg;
            end
            PGmax_new = max(PGmax_new, PG);
            PGmin_new = min(PGmin_new, PG);
            if abs(PG)>1e-12
                betaold(i) = beta(i);
                beta(i) = min(max(beta(i)-pg/H_Diag(i), 0), C_eORs(i)/m);
                u = u + (beta(i)-betaold(i))*invQKY(:, i);
            end
        end
        
        X_old = X_new;
        
        if  PGmax_new-PGmin_new<=eps_bound
            if length(X_old) == m
                break;
            else
                X_old = 1:m;  X_new = 1:m;
                PGmax_old = inf;   PGmin_old = -inf;
            end
        end
        
        if  PGmax_new<=0
            PGmax_old = inf;
        else
            PGmin_old = PGmax_new;
        end
        
        if  PGmin_old>=0
            PGmin_old = -inf;
        else
            PGmin_old = PGmin_new;
        end
        iter = iter+1;  
    end
end
end
