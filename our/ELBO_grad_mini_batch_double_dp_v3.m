%ind
%events
%
%model.U, the latent factors stored in each mode
%model.tau, 
%model.T, the time interval
%batch_indices, the indices of the batch samples
%data.ind, data.e, data.par, data. child
%mode.subs: all the tensor entries

%parent: the indcies of parent candidate for each event
%children: the indicies child candidate for each event
%double means there is another batch

%%%%
%compared with v2, add different calculation for integration window-based triggering
%function
function [df, model] =  ELBO_grad_mini_batch_double_dp_v3(x, data, batch_indices, batch_entries, model)
    %deassemble
    %1) psuedo input
    d = model.pseudo_dim; %dimension
    m = model.np; %# of pseudo inputs
    dim = model.dim; %#ranks
    nmod = model.nmod;
    nvec = model.nvec;
    lam = model.lam;
    T = model.T;
    N = length(data.e);
    Ne = size(model.subs,1);
    %get event count for each entry in the mini-batch
    e_sub_count = sptensor(data.ind(batch_indices(1),:), 1, nvec);
    n_events = length(batch_indices);
    n_entries = length(batch_entries);
    for i=2:n_events
        sub = data.ind(batch_indices(i),:);
        e_sub_count(sub) = e_sub_count(sub) + 1;
    end
    %entry indices
    subs = find(e_sub_count);
    subs2entry_ind = model.subs2index(subs);
    n_subs  = size(subs,1);
    [U, Um, mu, L, ker_param, a, b, tau] = decode_parameters(x, model);
    S = L*L';
    sigma = ker_param.sigma;
    sigma0 = ker_param.sigma0;
    
    st = 0;
    X = zeros(n_subs, d);
    for k=1:nmod
        X(:,st+1:st+dim(k)) = U{k}(subs(:,k),:);
        st = st + dim(k);
    end

    dummy = ones(size(nvec));
    thetas = sptensor(dummy, 1, nvec);
    T_tilt = sptensor(subs, e_sub_count(subs)./model.e_count(subs),  nvec);
    
    
    Kmm = ker_func(Um, ker_param);
    Knm = ker_cross(X, Um, ker_param);
    KmmInv = Kmm^(-1);
    KmmInvL = Kmm\L;
    KmmInvKmn = Kmm\Knm';
    KnmKmmInvL = Knm*KmmInvL;
    KmmInvMu = Kmm\mu;
    KmmInvS = Kmm\S;
    KnmKmmInvMu = Knm*KmmInvMu;
    
    e_sub_vals = sptensor(subs, KnmKmmInvMu, nvec);
    
    %starting from here
    %need to calc theta, first, and then use the code for latent factors
    gU = cell(nmod,1);
    for k=1:nmod
        %might need change in the future (e.g., do not regularize latent factros & pseudo inputs)
        %gU{k} = zeros(size(U{k})) - U{k};
        gU{k} = zeros(size(U{k}));%this works better
    end
    z = cell(n_events,1);
    ExptLogBta = psi(a) - log(b);
    ExptBta = a/b;
    for i=1:n_events
        event_index = batch_indices(i);
        z{i} = zeros(length(data.par{event_index}) + 1, 1);
        sub = data.ind(event_index,:);
        if length(z{i})==1
            z{i}(1) = 1;
        else
            %fprintf('i=%d\n',i);
            %if i==8
            %    fprintf('wocao\n');
            %end
            z{i}(1) = e_sub_vals(sub);
            z{i}(2:end) = ExptLogBta ...
                + get_dist_log(U, dim, d, lam, sub, data.ind(data.par{event_index},:)) ...
                - 1.0/tau*(data.e(event_index) - data.e(data.par{event_index}));
            z{i} = exp(z{i} - logsumexp(z{i}));
        end 
        thetas(sub) = thetas(sub) + z{i}(1); %prob. of being triggered by background rate           
    end
    thetas(dummy) = thetas(dummy) - 1;
    
    %natural parameters for q(bta) & g_tau 
    a_new = model.a0;
    b_new = model.a1;
    g_tau = 1/tau*(model.b0-1) - model.b1;
    for i=1:n_events
        event_index = batch_indices(i);
        sn = data.e(event_index);
        sub = data.ind(event_index,:);
        sub_par = data.ind(data.par{event_index},:);
        if ~isempty(sub_par)
            diff_par = get_diff(U, dim, d, sub, sub_par);
        end        
        %one event has potential influence over all the tensor entries
        %here we sample a batch entries
        %sub_child = model.subs;
        sub_child = model.subs(batch_entries,:);
        if ~isempty(sub_child)
            diff_child = get_diff(U, dim, d, sub, sub_child);
            child_dist = exp( get_dist_log(U, dim, d, lam, sub, sub_child) );
        end
        %window size only
        switch model.triggering_strategy
            case 'window'
                delta = min(model.Dmax, T-sn)/tau;
            case 'maxK'
                delta = (data.e(min(event_index+model.Kmax, N)) - sn)/tau;
            case 'hybrid'
                delta = (min(min(model.Dmax + sn, T), data.e(min(event_index+model.Kmax, N))) - sn)/tau;
        end
        %Nearest K
        %delta = (data.e(min(event_index+model.Kmax, N)) - sn)/tau;
        %Window size & Nearest K
        %delta = (min(min(model.Dmax + sn, T), data.e(min(event_index+model.Kmax, N))) - sn)/tau;
        a_new = a_new + N/n_events*(1-z{i}(1));
        if ~isempty(sub_child)
            b_new = b_new + (N/n_events)*(Ne/n_entries)*tau*(1-exp(-delta)) ...
                *sum(child_dist);
        end
        if ~isempty(sub_child)
            g_tau = g_tau - (N/n_events)*(Ne/n_entries)*ExptBta*(1 - (1+delta)*exp(-delta))*sum(child_dist);
        end
        if ~isempty(sub_par)
            g_tau = g_tau + N/n_events/(tau*tau)*sum((sn - data.e(data.par{event_index})).*z{i}(2:end));
        end
        
        %grad w.r.t latent factors
        sub2entry_ind = model.subs2index(sub);
        if ~isempty(sub_par)
            sub_par2entry_ind = model.subs2index(sub_par);
        end
        if ~isempty(sub_child)
            sub_child2entry_ind = model.subs2index(sub_child);
        end
        part1 = zeros(1,d);
        if ~isempty(sub_child)
            part1 = (N/n_events)*(Ne/n_entries)*ExptBta*tau*(1-exp(-delta))*2/lam...
                *(repmat(child_dist,1,d).*diff_child);
        end
        part2 = zeros(1,d);
        if ~isempty(sub_par)
            part2 = N/n_events*(-2/lam)*(repmat(z{i}(2:end), 1, d).*diff_par);
        end
        st = 0;
        for k=1:nmod
           part = sum(part1,1) + sum(part2,1);
           gU{k} = gU{k} + model.ind2entry{k}(:, sub2entry_ind)*part(:,st+1:st+dim(k));
           if ~isempty(sub_child)
                gU{k} = gU{k} + model.ind2entry{k}(:, sub_child2entry_ind)*(-part1(:,st+1:st+dim(k)));
           end
           if ~isempty(sub_par)
                gU{k} = gU{k} + model.ind2entry{k}(:, sub_par2entry_ind)*(-part2(:,st+1:st+dim(k)));
           end
           st = st + dim(k);
        end
    end
    g_log_tau = g_tau*tau;
    
    %added for testing, forbid updaing tau any more
%     if tau<0.02
%         g_log_tau = 0;
%     end

    h = exp(0.5*(sigma0+sigma - sum(Knm.*KmmInvKmn',2) + sum(KnmKmmInvL.^2,2)) + KnmKmmInvMu);
    h = (T_tilt(subs).*h)*N/n_events;
    KmmInvKmnDiagH = KmmInvKmn.*repmat(h', m, 1);
    y = thetas(subs)*N/n_events;    
    %gradident w.r.t GP
    g_mu = -KmmInvMu + KmmInvKmn*(y - T*h);
    g_L = diag(1./diag(L)) + tril(- KmmInvL - T*KmmInvKmn*(repmat(h,1,m).*KnmKmmInvL));
    %dKmm
    A1 = -0.5*KmmInv + 0.5*KmmInv*(mu*mu'+S)*KmmInv - KmmInvKmn*(y-T*h)*KmmInvMu' ...
        + T*(-0.5*eye(m) + KmmInvS)*(KmmInvKmnDiagH*KmmInvKmn');
    [g_Um1, g_kerPara1] = ker_grad(Um, A1, Kmm, ker_param);
    
    %dKmn
    A2 = KmmInvMu*y' + T*(eye(m) - KmmInvS)*KmmInvKmnDiagH - T*KmmInvMu*h' ;
    [g_Um2, g_kerParam2] = ker_cross_grad(A2', Knm', Um, X, ker_param);
    %g_Um = g_Um1 + g_Um2 - Um;
    g_Um = g_Um1 + g_Um2;
    g_log_l = g_kerPara1(1:d) + g_kerParam2(1:d);
    g_log_sigma = g_kerPara1(d+1) + g_kerParam2(d+1) -0.5*T*sum(h)*sigma;
    g_log_sigma0 = g_kerPara1(d+2) - 0.5*T*sum(h)*sigma0;
    [g_X, ~] = ker_cross_grad(A2, Knm, X, Um, ker_param);
    
    st = 0;
    df = vec(g_Um);
    for k=1:nmod
        gU{k} = gU{k} + model.ind2entry{k}(:, subs2entry_ind)*g_X(:,st+1:st+dim(k));
        st = st + dim(k);        
    end
    
    %regarding DP prior over latent factors    
    if isfield(model, 'dp')        
        model = update_dp_posterior_full(model, U);
        for k=1:nmod
            gU{k} = gU{k} - model.dp.lam(k)*(U{k} - model.dp.stat.phi{k}*model.dp.stat.eta_mean{k});
        end
    end
    
    for k=1:nmod
        df = [df; vec(gU{k})];
    end
    lower_ind = tril(ones(m))==1;
    df = [df; g_mu; g_L(lower_ind)];
    df = [df; g_log_l; g_log_sigma; g_log_sigma0; a_new - a ; b_new - b; g_log_tau];    
    if sum(isinf(df))>0 || sum(isnan(df))>0
        fprintf('inf or NaN!\n');
    end
end