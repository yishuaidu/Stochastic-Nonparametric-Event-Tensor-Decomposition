%support ARD kernel only, k(x,y) = \sigma exp(-\frac{1}{2} (x-y)^\top diag(1/\l_k)) + \sigma0
function K = ker_func(X, ker_param)
    n = size(X,1);
    if strcmp(ker_param.type, 'ard')
        sigma0 = ker_param.sigma0;
        sigma = ker_param.sigma;
        l = ker_param.l;
        X = X*diag(1./sqrt(l));
        P = sum(X.*X,2);
        K = sigma*exp(-0.5*(repmat(P',n,1) + repmat(P,1,n) - 2*(X*X'))) + (sigma0+ker_param.jitter)*eye(n);
    else
        error('UnSupported kernel type');
    end
end

