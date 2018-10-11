%
function [g] =  log_evidence_lower_bound_batch(x, model, sel)
    x = vec(x);
    nmod = length(model.nvec);
    nvec = model.nvec;
    U = cell(nmod,1);
    st = 0;
    for k=1:nmod
        U{k} = reshape(x(st + 1 : st + nvec(k)*model.R), nvec(k), model.R);
        st = st + nvec(k)*model.R;
    end
    %# of entries
    n = size(model.subs,1);
    n_batch = length(sel);
    y = model.y(sel)*n/n_batch;
    T = model.T *n/n_batch;
    nmod = model.nmod;    
    %ind_batch = model.subs(sel,:);
    U_batch = cell(nmod, 1);
    M = ones(n_batch, model.R);
    for k=1:nmod
        U_batch{k} = U{k}(model.subs(sel, k),:);
        M = M.*U_batch{k};
    end
    lam_batch = exp(sum(M,2));
    g_batch = cell(nmod,1);
    for k=1:nmod
        inner_held_out = M./U_batch{k};
        g_batch{k} = diag(-T*lam_batch + y)*inner_held_out;
    end
    gU = cell(nmod, 1);
    g = [];
    for k=1:nmod
        gU{k} = model.ind2entry{k}(:,sel)*g_batch{k} - U{k};
        g = [g(:); vec(gU{k})];
    end   
    if sum(isinf(g)) || sum(isnan(g))
        fprintf('inf!\n');
    end
end