%initializatoin: pseudo input
%training from here)
%data.train_ind, the 
%model.U, latent factors
%model.dim, a vector, i-th entry, the rank in dim i
%model.np, # of pseudo inputs
%support ARD kernel only, right now
function model = do_init(model, data)
    nvec = data.tensor_sz;
    nmod = length(nvec);
    model.nvec = nvec;
    model.nmod = nmod;
    model.subs = data.train_subs;
    model.y = data.y_subs;
    %latent factor indices --> entry indices --> necessary for gradient
    %calculation w.r.t latent factors
    model.ind2entry = cell(nmod,1);    
    n = size(model.subs, 1);%# of observed entries
    for k=1:nmod
        model.ind2entry{k} = zeros(nvec(k), n);
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
    %model.Um = Xt(randsample(size(model.subs,1),model.np),:);
    model.pseudo_dim = size(model.Um, 2);
    model.oldUm = model.Um;
    
end