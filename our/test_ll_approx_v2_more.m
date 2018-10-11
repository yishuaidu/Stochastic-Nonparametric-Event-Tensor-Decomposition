%data.ind, indicies of all the events
%data.e, time of all the events
%this is not ELBO, instead an approx. to LL
%supports all kinds of truncated triggering kernels
%this is simply to verify my analysis
%n_set, the number of segments I want to test separately, and also return
%the whole
function res = test_ll_approx_v2_more(model, data, n_seg)
    d = model.pseudo_dim; %dimension
    dim = model.dim; %#ranks
    nmod = model.nmod;
    nvec = model.nvec;
    %Dmax = model.Dmax;
    test_indices = (1:length(data.e))';
    T1 = min(data.e(test_indices));
    T2 = max(data.e(test_indices));
    data.e = data.e';
    
    per_seg_event_no = ceil(length(data.e)/n_seg);
    res = zeros(n_seg,1);
    
    
    %get event count for each entry in the mini-batch
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
    %h = <\lambda_r^0> = <exp(f(x_r))>    
    h = exp(0.5*(ker_param.sigma0+ker_param.sigma - sum(Knm.*KmmInvKmn',2) + sum(KnmKmmInvL.^2,2)) + KnmKmmInvMu);
    e_sub_vals = sptensor(subs, log(h), nvec);
    %e_sub_vals = sptensor(subs, KnmKmmInvMu, nvec);
    
    ExptBta = a/b;
    for k=1:n_seg
        n_events = min(k*per_seg_event_no, length(test_indices));
        T1 = data.e((k-1)*per_seg_event_no + 1);
        T2 = data.e(n_events);
        res(k) = 0;
        for i=(k-1)*per_seg_event_no+1:n_events
            event_index = test_indices(i);
            sn = data.e(event_index);
            sub = data.ind(event_index,:);
    %         sub_par = data.ind(data.par{event_index},:);
    %         if ~isempty(sub_par)
    %             par_dist_log = get_dist_log(U, dim, d, lam, sub, sub_par);
    %         end        
    %         sub_child = data.ind(data.child{event_index},:);
    %         if ~isempty(sub_child)
    %             child_dist = exp( get_dist_log(U, dim, d, lam, sub, sub_child) );
    %         end

            dist2all_entries = exp(get_dist_log(U, dim, d, lam, sub, subs));
            %delta = min(T2-sn, Dmax)/tau;
            switch model.triggering_strategy
                case 'window'
                    delta = min(model.Dmax, T2-sn)/tau;
                case 'maxK'
                    delta = (data.e(min(event_index+model.Kmax, n_events)) - sn)/tau;
                case 'hybrid'
                    delta = (min(min(model.Dmax + sn, T2), data.e(min(event_index+model.Kmax, n_events))) - sn)/tau;
            end

            res(k) = res(k) - ExptBta*tau*(1-exp(-delta))*sum(dist2all_entries);
            terms = zeros(length(data.par{event_index}) + 1, 1);
            terms(1) = e_sub_vals(sub);
            if length(terms)>1
                terms(2:end) = log(ExptBta) ...
                    + get_dist_log(U, dim, d, lam, sub, data.ind(data.par{event_index},:)) ...
                    - 1.0/tau*(data.e(event_index) - data.e(data.par{event_index}));
            end
            res(k) = res(k) + logsumexp(terms);        
        end
        res(k) = res(k) - (T2 - T1)*sum(h);
    end
    
    %res = res - (T2 - T1)*sum(h);
    
end