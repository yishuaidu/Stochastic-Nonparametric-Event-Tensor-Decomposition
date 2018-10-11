%pre-processing, initilaize psuedo input using kmeans++
%data.ind, data.e
function model = do_init(model, data)
   model.nmod = size(data.ind, 2);
   model.nvec = zeros(1, model.nmod);
   for k=1:model.nmod
        model.nvec(k) = max(data.ind(:,k));%length(unique(data.ind(:,k)));
   end
   %even count for each entry
   model.e_count = sptensor(data.ind(1,:), 1, model.nvec);
   for n=2:size(data.ind, 1)
        sub = data.ind(n,:);
        model.e_count(sub) = model.e_count(sub) + 1;
   end
   %distinct entry indices
   model.subs = find(model.e_count);
   n = size(model.subs, 1);    
   model.subs2index = sptensor(model.subs, (1:n)', model.nvec);
   
    %latent factor indices --> entry indices --> necessary for gradient
    %calculation w.r.t latent factors
    nmod = model.nmod;
    nvec = model.nvec;
    model.ind2entry = cell(nmod,1);    
    for k=1:nmod
        model.ind2entry{k} = sparse(nvec(k), n);
        for j=1:nvec(k)
            model.ind2entry{k}(j,:) = (model.subs(:,k) == j)';
        end        
    end
    
    
    %kmeans for setting up pseudo inputs
    Xt = zeros(size(model.subs,1), sum(model.dim));
    st = 0;
    for k=1:nmod
        Xt(:, st+1:st+model.dim(k)) = model.U{k}(model.subs(:,k),:);
        st = st + model.dim(k);
    end
    [~,model.Um] = fkmeans(Xt', model.np);
    model.Um = model.Um';
    model.pseudo_dim = size(model.Um, 2);
    model.oldUm = model.Um;
    
    mdist = median(pdist(model.Um));
    d = model.pseudo_dim;
    if mdist>0
        log_l = 2*log(mdist)*ones(d,1);
    else
        log_l = zeros(d,1);
    end
    log_sigma = 0;
    %log_sigma0 = log(1e-6);
    log_sigma0 = log(1e-3); %best for this simulation
    ker_param = [];
    ker_param.type = 'ard';
    ker_param.l = exp(log_l);
    ker_param.sigma = exp(log_sigma);
    ker_param.sigma0 = exp(log_sigma0);
    ker_param.jitter = 1e-10;
    model.ker_param = ker_param;    
    model.mu = zeros(model.np,1);
    model.L = eye(model.np);
    
    %use nvecs as the initializaiton
    if strcmp(model.init_opt, 'nvecs')
        %log_count = tenfun(@(x) log(x+1), tensor(model.e_count/model.T));
        log_count = model.e_count/model.T;
        for k=1:model.nmod
            model.U{k} = nvecs(log_count, k, model.dim(k));
        end
    else
        if strcmp(model.init_opt, 'gp-pp')
            gppp = load('../../RFP-PP/model-gp-pp.mat');
            model.ker_param = gppp.model.ker_param;
            model.Um = gppp.model.Um;
            model.U = gppp.model.U;
            model.oldU = gppp.model.U;
            model.oldUm = model.Um;
            model.mu = gppp.model.mu;
            model.L = gppp.model.L;
            model.oldmu = model.mu;
            model.oldL = model.L;
        end
    end
    
    if ~isfield(model, 'triggering_strategy')
        model.triggering_strategy = 'Kmax';
        model.Kmax = 100;
    end
    
    %%allow DP inference
    if isfield(model, 'dp')
        %for variational posteriors/statistics    
        if ~isfield(model.dp, 'T')
            model.dp.T = round(nvec/10);    
        end
        if ~isfield(model.dp, 'lam')
            model.dp.lam = 1.0*ones(nmod,1);
        else
            %take inverse for convenience
            model.dp.lam= 1./model.dp.lam;
        end
        if ~isfield(model.dp, 's')
            model.dp.s = 1.0*ones(nmod,1);
        else
            %take inverse for convenience
            model.dp.s = 1./model.dp.s;
        end
        if ~isfield(model.dp, 'var_iter')
            model.dp.var_iter = 50;
        end
        model.dp.alpha = 1.0;
        model.dp.ga = cell(nmod,1);
        model.dp.stat = [];
        
        %expectation of log(v) and log(1-v)
        model.dp.stat.ex_logv = cell(nmod,1);
        %parameters for cluster membership
        model.dp.stat.phi = cell(nmod,1);
        %parameters for cluster centers
        model.dp.stat.eta_mean = cell(nmod,1);
        model.dp.stat.eta_cov = cell(nmod,1);
        %expt for norm^2
        model.dp.stat.eta_norm2 = cell(nmod,1);
        
        for k=1:nmod
            trunc_no = model.dp.T(k);
            model.dp.ga{k} = zeros(trunc_no, 2);
            model.dp.stat.ex_logv{k} = zeros(trunc_no, 2);
            %model.dp.stat.phi{k} = drchrnd(ones(1, trunc_no),nvec(k));
            model.dp.stat.phi{k} = 1/trunc_no*ones(nvec(k), trunc_no);
            model.dp.stat.eta_mean{k} = rand(trunc_no, model.dim(k));
            %model.dp.stat.eta_mean{k} = randn(trunc_no, model.dim(k));
            model.dp.stat.eta_cov{k} = 1/model.dp.lam(k)*ones(1, trunc_no);
            model.dp.stat.eta_norm2{k} = model.dim(k)*model.dp.stat.eta_cov{k} + sum(model.dp.stat.eta_mean{k}.^2,2)';
        end
        %create the cache stats
        model.dp.cache_stat = [];
        %\sum \Phi_k^{n,t}
        model.dp.cache_stat.psy1 = cell(nmod,1);
        %\sum \Phi_k^{n,t}U_k^n
        model.dp.cache_stat.psy2 = cell(nmod,1);
        %\sum_{n=1}^m_k \sum_{j=t+1}^T \Phi_k^{n,j}
        model.dp.cache_stat.psy3 = cell(nmod,1);
        for k=1:nmod
            model.dp.cache_stat.psy1{k} = sum(model.dp.stat.phi{k}, 1);
            model.dp.cache_stat.psy2{k} = model.dp.stat.phi{k}'*model.U{k};
            model.dp.cache_stat.psy3{k} = zeros(1, model.dp.T(k));
            csum = cumsum(fliplr(model.dp.cache_stat.psy1{k}));
            csum = fliplr(csum(1:end-1));
            model.dp.cache_stat.psy3{k} = [csum,0];                                
        end
         
    end
   
end