function ll = predictive_log_ELBO(model, dt)
    %essentially is a ELBO
    X = zeros(size(dt.test_ind,1), sum(model.dim));
    st = 0;
    for k=1:model.nmod
        X(:,st + 1:st + model.dim(k)) = model.U{k}(dt.test_ind(:,k),:);
        st= st + model.dim(k);
    end
    ker_param = model.ker_param;
    Um = model.Um;
    L = model.L;
    mu = model.mu;
    
    
    
    Kmm = ker_func(Um, ker_param);
    Knm = ker_cross(X, Um, ker_param);
    KmmInvL = Kmm\L;
    KmmInvKmn = Kmm\Knm';
    KnmKmmInvL = Knm*KmmInvL;
    KmmInvMu = Kmm\mu;
    KnmKmmInvMu = Knm*KmmInvMu;
    h = exp(0.5*(ker_param.sigma0 + ker_param.sigma - sum(Knm.*KmmInvKmn',2) + sum(KnmKmmInvL.^2,2)) + KnmKmmInvMu);
    lam = KnmKmmInvMu;
    y = dt.test_vals;
    %ll = sum(-dt.T*h + y.*(lam + log(dt.test_T)) - gammaln(y));    
    ll = sum(-dt.test_T*h + y.*lam); 
    %ll = sum(y.*log(lam) - lam - gammaln(y));
end