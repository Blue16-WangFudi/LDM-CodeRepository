function U = Fuzzy_MemberShip_FCM(data, label)
%--------------------------------
cluster_n = 1;
 
data_p = data(label==1,:);
data_n = data(label==-1,:);
center_p = fcm(data_p , 1); % 聚类中心
center_n = fcm(data_n , 1);
dist_pp = distfcm(center_p, data_p);       % 计算距离矩阵
dist_nn = distfcm(center_n, data_n);
dist_np = distfcm(center_p, data_n);       % 计算距离矩阵
dist_pn = distfcm(center_n, data_p);
% if (dist_pp ==0)
%     U_p_p = 1;
% else if dist_nn == 0
%         U_n_n = 1;
%     else
U_p_p = (dist_pp.^(-2))./(ones(cluster_n, 1)*(dist_pn.^(-2))+(dist_pp.^(-2)));  % 计算最终的隶属度矩阵  p类
U_p_n = ones(1,length(U_p_p)) - U_p_p;
U_n_n = (dist_nn.^(-2))./(ones(cluster_n, 1)*(dist_np.^(-2))+(dist_nn.^(-2)));  % 计算最终的隶属度矩阵  n类
U_n_p = ones(1,length(U_n_n)) - U_n_n;
%         if dist_pp ==0
%             U_p_p(find(dist_pp==0))=1;
%         elseif dist_nn ==0
%             U_n_n(find(dist_nn==0))=1;
%         end
%U = [U_p_p U_n_p;U_p_n U_n_n];
%     end
U_t = [U_p_p U_n_n]';
for i=1:length(U_t)
    if isnan(U_t(i))
        U_t(i)=1;
    end
end
U=U_t;




end

