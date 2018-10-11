%initializatoin: pseudo input
%training from here)
%data.train_ind, the 
%model.U, latent factors
%model.dim, a vector, i-th entry, the rank in dim i
%model.np, # of pseudo inputs
function model = do_init(model, data)
    nvec = data.tensor_sz;
    nmod = length(nvec);
    model.nvec = nvec;
    model.nmod = nmod;
    %entry indices
    model.subs = data.train_subs;
    %count of events in each observed entry
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
end