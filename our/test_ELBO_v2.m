%data.ind, indicies of all the events
%data.e, time of all the events
%this is actually the variational ELBO
%support all kind of truncated triggering kernels
function res = test_ELBO_v2(model, data, test_indices)
    d = model.pseudo_dim; %dimension
    dim = model.dim; %#ranks
    nmod = model.nmod;
    nvec = model.nvec;
    T1 = min(data.e(test_indices));
    T2 = max(data.e(test_indices));
    
    %get event count for each entry in the test-batch
    e_sub_count = sptensor(data.ind(test_indices(1),:), 1, nvec);
    n_events = length(test_indices);
    for i=2:n_events
        sub = data.ind(test_indices(i),:);
        e_sub_count(sub) = e_sub_count(sub) + 1;
    end
    %entry indices
    subs = find(e_sub_count);
    n_subs  = size(subs,1);
    U = model.U;
    Um = model.Um;
    mu = model.mu;
    L = model.L;
    ker_param = model.ker_param;
    a = model.a;
    b = model.b;
    lam = model.lam;
    tau = model.tau;
    %S = L*L';
    
    X = zeros(n_subs, d);
    st = 0;
    for k=1:nmod
        X(:,st+1:st+dim(k)) = U{k}(subs(:,k),:);
        st = st + dim(k);
    end

    
    
    Kmm = ker_func(Um, ker_param);
    Knm = ker_cross(X, Um, ker_param);
    %KmmInv = Kmm^(-1);
    KmmInvL = Kmm\L;
    KmmInvKmn = Kmm\Knm';
    KnmKmmInvL = Knm*KmmInvL;
    KmmInvMu = Kmm\mu;
    %KmmInvS = Kmm\S;
    KnmKmmInvMu = Knm*KmmInvMu;    
    e_sub_vals = sptensor(subs, KnmKmmInvMu, nvec);
    
    z = cell(n_events,1);
    ExptLogBta = psi(a) - log(b);
    ExptBta = a/b;
    res = 0;
    for i=1:n_events
        event_index = test_indices(i);
        sn = data.e(event_index);
        sub = data.ind(event_index,:);
        sub_par = data.ind(data.par{event_index},:);
        if ~isempty(sub_par)
            par_dist_log = get_dist_log(U, dim, d, lam, sub, sub_par);
        end        
        %sub_child = data.ind(data.child{event_index},:);
        sub_child = subs; %data.ind(data.child{event_index},:);
        if ~isempty(sub_child)
            child_dist = exp( get_dist_log(U, dim, d, lam, sub, sub_child) );
        end
        
        z{i} = zeros(length(data.par{event_index}) + 1, 1);        
        if length(z{i})==1
            z{i}(1) = 1;
        else
            z{i}(1) = e_sub_vals(sub);
            z{i}(2:end) = ExptLogBta ...
                + get_dist_log(U, dim, d, lam, sub, data.ind(data.par{event_index},:)) ...
                - 1.0/tau*(data.e(event_index) - data.e(data.par{event_index}));
            log_p = z{i} - logsumexp(z{i});
            z{i} = exp(log_p);
            %entropy for q(z{i})
            res = res - sum(log_p.*z{i});
        end 
        %delta = (T2-sn)/tau;
        %delta = min(model.Dmax, T2-sn)/tau;
        switch model.triggering_strategy
            case 'window'
                delta = min(model.Dmax, T2-sn)/tau;
            case 'maxK'
                delta = (data.e(min(event_index+model.Kmax, n_events)) - sn)/tau;
            case 'hybrid'
                delta = (min(min(model.Dmax + sn, T2), data.e(min(event_index+model.Kmax, n_events))) - sn)/tau;
        end

        
        if ~isempty(sub_child)
            res = res - ExptBta*tau*(1-exp(-delta))*sum(child_dist);
        end
        res = res + e_sub_vals(sub)*z{i}(1);
        if ~isempty(sub_par)
            res = res + sum(z{i}(2:end).*(ExptLogBta + par_dist_log - 1/tau*(sn - data.e(data.par{event_index}))));
        end        
    end
    h = exp(0.5*(ker_param.sigma0+ker_param.sigma - sum(Knm.*KmmInvKmn',2) + sum(KnmKmmInvL.^2,2)) + KnmKmmInvMu);
    res = res - (T2 - T1)*sum(h);
    
end