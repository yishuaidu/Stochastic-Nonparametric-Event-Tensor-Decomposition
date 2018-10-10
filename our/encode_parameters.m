function x = encode_parameters(model)
    x = model.Um(:);
    for k=1:model.nmod
         x = [x; vec(model.U{k})];
    end
    lower_ind = tril(ones(model.np))==1;
    x = [x; model.mu; model.L(lower_ind)];
    log_sigma = log(model.ker_param.sigma);
    log_sigma0 = log(model.ker_param.sigma0);
    log_l = log(model.ker_param.l);
    x = [x;log_l;log_sigma;log_sigma0];
    x = [x;model.a; model.b; log(model.tau)];
end