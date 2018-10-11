function ll = predictive_log_likelihood(model, data)
    ntest = size(data.test_ind,1);
    M = ones(ntest, model.R);
    for k=1:model.nmod
        M = M.*model.U{k}(data.test_ind(:,k),:);
    end
    lam = exp(sum(M,2));
    T = data.test_T;
    y = data.test_vals;
    ll = sum(y.*log(lam) - lam*T);
end