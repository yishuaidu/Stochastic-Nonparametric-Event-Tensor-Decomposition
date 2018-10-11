%
function [f, df] =  log_evidence_lower_bound_batch(x, model, sel)
    x = vec(x);
    nmod = length(model.nvec);
    nvec = model.nvec;
    %deassemble
    %1) psuedo input
    d = model.pseudo_dim; %dimension
    m = model.np; %# of pseudo inputs
    dim = model.dim; %#ranks
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
    S = L*L';
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
    n = size(model.subs,1);
    n_batch = length(sel);
    y = model.y(sel)*n/n_batch;
    T = model.T *n/n_batch;
    X = zeros(n_batch, d);
    nmod = model.nmod;
    
    st = 0;
    for k=1:nmod
        X(:,st+1:st+model.dim(k)) = U{k}(model.subs(sel,k),:);
        st = st + dim(k);
    end
    
    Kmm = ker_func(Um, ker_param);
    Knm = ker_cross(X, Um, ker_param);
    KmmInv = Kmm^(-1);
    KmmInvL = Kmm\L;
    KmmInvKmn = Kmm\Knm';
    KnmKmmInvL = Knm*KmmInvL;
    KmmInvMu = Kmm\mu;
    KmmInvS = Kmm\S;
    KnmKmmInvMu = Knm*KmmInvMu;
    h = exp(0.5*(sigma0+sigma - sum(Knm.*KmmInvKmn',2) + sum(KnmKmmInvL.^2,2)) + KnmKmmInvMu);
    KmmInvKmnDiagH = KmmInvKmn.*repmat(h', m,1);
    
    f = -0.5*logdet(Kmm) - 0.5*trace(Kmm\(mu*mu' + S)) + sum(KnmKmmInvMu.*y) ...
        -T*sum(h) + sum(log(diag(L)));
    %prior for psuedo input & latent factors
    f = f - 0.5*sum(vec(Um.^2));
    for k=1:nmod
        f = f - 0.5*sum(vec(U{k}.^2));
    end
    %gradident part
    g_mu = -KmmInvMu + KmmInvKmn*(y - T*h);
    g_L = diag(1./diag(L)) + tril(- KmmInvL - T*KmmInvKmn*(repmat(h,1,m).*KnmKmmInvL));
    %dKmm
    A1 = -0.5*KmmInv + 0.5*KmmInv*(mu*mu'+S)*KmmInv - KmmInvKmn*(y-T*h)*KmmInvMu' ...
        + T*(-0.5*eye(m) + KmmInvS)*(KmmInvKmnDiagH*KmmInvKmn');
    [g_Um1, g_kerPara1] = ker_grad(Um, A1, Kmm, ker_param);
    
    %dKmn
    A2 = KmmInvMu*y' + T*(eye(m) - KmmInvS)*KmmInvKmnDiagH - T*KmmInvMu*h' ;
    [g_Um2, g_kerParam2] = ker_cross_grad(A2', Knm', Um, X, ker_param);
    g_Um = g_Um1 + g_Um2 - Um;
    g_log_l = g_kerPara1(1:d) + g_kerParam2(1:d);
    g_log_sigma = g_kerPara1(d+1) + g_kerParam2(d+1) -0.5*T*sum(h)*sigma;
    g_log_sigma0 = g_kerPara1(d+2) - 0.5*T*sum(h)*sigma0;
    [g_X, ~] = ker_cross_grad(A2, Knm, X, Um, ker_param);
    gU = cell(nmod,1);
    st = 0;
    df = vec(g_Um);
    for k=1:nmod
        gU{k} = model.ind2entry{k}(:,sel)*g_X(:,st+1:st+dim(k)) - U{k};
        st = st + dim(k);
        df = [df; vec(gU{k})];        
    end
    df = [df; g_mu; g_L(lower_ind)];
    df = [df; g_log_l; g_log_sigma; g_log_sigma0];
    if isinf(f) || isnan(f)
        fprintf('inf!\n');
    end
end