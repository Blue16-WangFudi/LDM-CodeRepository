function  [alpha, beta] = CD_LDM(Q, GY, lambda2, C_s, eps, max_iter)
% Function:  Coordinate Descent method for the dual problem of LDM
%------------------- Input -------------------%
% Q, P(=YG), lambda2, C_s, eps, max_iter

%------------------- Output -------------------%
% alpha, beta


% Author: Wendong Wang
%  Email: d.sylan@foxmail.com
%   data: 2015,9,8

% This is the first version of our CD_LDM algorithm. One can find more in following webs
% 1. http://lamda.nju.edu.cn/gaobb/
% 2. http://github.com/gaobb/CDFTSVM


 rng('default') 

if ( nargin>6||nargin<4) % check correct number of arguments
    help CD_LDM
else
    if (nargin<5)
        eps = 1e-3;
    end
    if (nargin<6)
        max_iter = 200;
    end
    
%     H_Diag = diag(GY'*(Q\GY));
%     invQGY = Q\GY;
    
    H_Diag = diag(GY'*inv(Q)*GY);
    invQGY = inv(Q)*GY;
    
    m =size(GY, 2);   
    X_new = 1:m;
    X_old = 1:m;
    
    beta  = zeros(m, 1); 
    betaold = zeros(m, 1);
%     alpha = lambda2*Q\(GY*ones(m, 1))/m; 
    alpha = lambda2*inv(Q)*GY*ones(m, 1)/m; 
    
    PGmax_old = inf;       %M_bar
    PGmin_old = -inf;      %m_bar
    
    iter = 0;    
    while iter<max_iter
        PGmax_new = -inf;   %M
        PGmin_new = inf;   %m
        R = length(X_old);
        X_old = X_old(randperm(R));
        
        for  j = 1:R
            i = X_old(j);
            pg = GY(:, i)'*alpha-1;  
            PG = 0;               
            if beta(i) == 0
                if pg>PGmax_old
                    X_new(X_new==i) = [];
                    continue;
                elseif  pg<0
                    PG = pg;
                end
            elseif beta(i)==C_s(i)/m
                if pg<PGmin_old
                    X_new(X_new==i) = [];
                    continue;
                elseif  pg>0
                    PG = pg;
                end
            else
                PG = pg;
            end
            PGmax_new = max(PGmax_new,PG);
            PGmin_new = min(PGmin_new,PG);
            if abs(PG)> 1e-12
                betaold(i) = beta(i);
                beta(i) = min(max(beta(i)-pg/H_Diag(i), 0.0), C_s(i)/m);
                alpha = alpha + (beta(i)-betaold(i))*invQGY(:, i);
            end
        end
        
        X_old = X_new;
        
        if  PGmax_new-PGmin_new<=eps
            if length(X_old)==m
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
