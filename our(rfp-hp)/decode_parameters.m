%extract the parameters from the vector x
%latent factors, pseudo inputs, psoterior mean and covariance for inducing
%targets,kernel parameters, natural parameters for bta in triggering
%kernel, lenght scale in triggering kernel
function [U, Um, mu, L, ker_param, a, b, tau] = decode_parameters(x, model)
    x = vec(x);
    nmod = length(model.nvec);
    nvec = model.nvec;
    
    %1) psuedo input
    d = model.pseudo_dim; %dimension
    m = model.np; %# of pseudo inputs
    xp = x(1 : m*d);
    Um = reshape(xp, m, d);
    st = m*d;
    U = cell(nmod,1);
    %2) latent factors
    for k=1:nmod
        U{k} = reshape(x(st + 1 : st + nvec(k)*model.dim(k)), nvec(k), model.dim(k));
        st = st + nvec(k)*model.dim(k);
    end
    %3) variational posterios for pseudo targets
    mu = x(st+1 : st+m);
    st = st + m;
    L = zeros(m);
    lower_ind = tril(ones(m))==1;
    L(lower_ind) = x(st+1 : st+m*(m+1)/2);
    st = st + m*(m+1)/2;
    %4) kernel parameters (we optimize them in log domain to ensure
    %postiveness)
    l = exp(x(st+1 : st+d));
    sigma = exp(x(st+d+1));
    sigma0 = exp(x(st+d+2));
    ker_param = [];
    ker_param.type = 'ard';
    ker_param.l = l;
    ker_param.sigma = sigma;
    ker_param.sigma0 = sigma0;
    ker_param.jitter = 1e-10;
    st = st + d + 2;
    %a, b, natural parameters q(bta)
    a = x(st+1);
    b = x(st+2);
    %5)tau, length scale for trigger kernel in Hawkes process
    tau = exp(x(st+3));
  
end