%---data----
%indices for training
%data.train_subs
%data.y_subs;
%data.tensor_sz;
%data.T, the time period
%---model---
%model.nvec: the latent dimension for each mode
%model.np: number of pseudo inputs
%model.Um: m*(d1+...+dk), pseudo input
%model.U: latent factors {1}, ..., {K}
%model.mu: posterior mean of q(g) -- pseudo targets
%model.L: lower chol of covairance of q(g)
%GP tensor facotrization,each entry is a poisson point process
%save all the models
function [model_list, test_ll_app, test_elbo] = GPTensorPP_online_advanced(data, model)
    %initialization    
    model = do_init(model, data);
    %assemble
    nmod = model.nmod;
    x = model.Um(:);
    for k=1:nmod
         x = [x; vec(model.U{k})];
    end
    d = model.pseudo_dim;
    m = model.np;
    mu = zeros(m,1);
    L = eye(m);
    lower_ind = tril(ones(m))==1;
    x = [x; mu; L(lower_ind)];
    mdist = median(pdist(model.Um));
    if mdist>0
        log_l = 2*log(mdist)*ones(d,1);
    else
        log_l = zeros(d,1);
    end
    log_sigma = 0;
    log_sigma0 = log(1e-1);
    %log_sigma0 = -inf;
    x = [x;log_l;log_sigma;log_sigma0];    
    %fastDerivativeCheck(@(x) log_evidence_lower_bound_full(x,model), x);
   
    %opt = [];
    %opt.MaxIter = 100;
    %opt.MaxFunEvals = 10000;
    %new_x = minFunc(@(x) log_evidence_lower_bound_full(x, model), x, opt);
    n = size(model.subs,1);
    decay = model.decay;
    batch_size = model.batch_size;
    epsi = 1e-5;
    step_rate = 1;
    sms = zeros(length(x), 1);
    gms = zeros(length(x), 1);
    samples = randperm(size(model.subs,1));
    epoch = model.epoch;
    test_ll_app = zeros(epoch,1);
    test_elbo = zeros(epoch, 1);
    model_list = cell(epoch, 1);
    for j=1:epoch
        st = 1;
        while st < n
            to_sel = samples(st:min(st+batch_size-1, n));
            [~, grad] = log_evidence_lower_bound_batch(x, model, to_sel);
            gms = decay * gms + (1 - decay) * grad.^2;
            step = sqrt(sms + epsi) ./ sqrt(gms + epsi) .* grad * step_rate;
            x = x + step;
            sms = decay * sms + (1 - decay) * step.^2;
            n_batch = ceil(st/batch_size);
            if mod(n_batch, 5) == 0
                fprintf('epoch = %d, batch %d\n', j, n_batch);
            end
            st = st + batch_size;
        end
        
        %deassemble
        model.Um = reshape(x(1:numel(model.Um)), size(model.Um));
        st = numel(model.Um);
        for k=1:model.nmod
            model.U{k} = reshape(x(st + 1 : st + numel(model.U{k})), size(model.U{k}));
            st = st + numel(model.U{k});
        end
        model.mu = x(st+1:st+m);
        model.L = zeros(m);
        model.L(lower_ind) = x(st + m + 1 : st + m + m*(m+1)/2);
        st = st + m + m*(m+1)/2;
        model.ker_param = [];
        model.ker_param.type = 'ard';
        model.ker_param.l = exp(x(st+1 : st+d));
        model.ker_param.sigma = exp(x(st+d+1));
        model.ker_param.sigma0 = exp(x(st+d+2));
        model.ker_param.jitter = 1e-10;
        model.Kmm = ker_func(model.Um, model.ker_param);
        
        test_ll_app(j) = predictive_log_likelihood_approx(model, data);
        test_elbo(j) = predictive_log_ELBO(model, data);
        %model_list{j} = model;
        model_list{j} = rmfield(model, {'subs','y', 'ind2entry'});
    end
    
    
    
end