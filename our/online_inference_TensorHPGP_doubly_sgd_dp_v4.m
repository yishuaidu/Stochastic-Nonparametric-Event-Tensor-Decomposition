%online inference for temproal tensor data, with GP and HP
%data.ind, the indices of the tensor entry associated with each event
%data.e, the time point for each event, we assume the time point are
%ordered
%model.U, latent factors
%model.ker_param, kernel parameters
%model.lam, used fro distance between tensor entries, exp(-\frac{1}{2\lam}
%||x - y||^2)
%model.tau, inverse scale for triggering kernel, note we use point
%estimation for tau
%model.b0, model.b1, parameters for gamma priors for model.tau 
%model.a0, model.a1, parameters for gamma priors for model.bta
%model.a, model.b, parameters for posteriors of model.bta (which is
%also a Gamma dist.)
%compared with v3, support different truncated triggering kernels
%every epoch
function [model, test_LL_approx, test_LL_ELBO, model_list] = online_inference_TensorHPGP_doubly_sgd_dp_v4(data, model, need_init, test_data, test_indices, file_name)
    nepoch = model.nepoch;
    test_LL_approx = zeros(nepoch,1);
    test_LL_ELBO = zeros(nepoch,1);
    model_list = cell(nepoch, 1);
    batch_size = model.batch_size;
    if need_init
        model = do_init(model, data);
    end
    N = length(data.e);
    x = encode_parameters(model);
    decay = model.decay;
    %decay2 = model.decay2; %for SVI for q(bta)
    epsi = 1e-5;
    step_rate = 1;
    sms = zeros(length(x), 1);
    gms = zeros(length(x), 1);
    for j=1:nepoch
       %ind = randperm(N);
       ind = 1:N;
       st = 1;
       while (st < N)
            %we can do random sampling as well
            if st+batch_size >= N
                to_sel = ind(st:N);
            else
                to_sel = ind(st:st + batch_size - 1);
            end
            to_sel_e = randsample(size(model.subs,1), 2*batch_size);
            %a = x(end-2);
            %b = x(end-1);
            %addelta            
            [grad, model] = ELBO_grad_mini_batch_double_dp_v3(x, data, to_sel, to_sel_e, model);
            gms = decay * gms + (1 - decay) * grad.^2;
            step = sqrt(sms + epsi) ./ sqrt(gms + epsi) .* grad * step_rate;
            x = x + step;
            sms = decay * sms + (1 - decay) * step.^2;
            %deal with a,b separately, natural parameters for q(bta)
            %a = decay2 * a + (1 - decay2) * grad(end-2);
            %b = decay2 * b + (1 - decay2) * grad(end-1);
            %x(end-2) = a;
            %x(end-1) = b;            
            n_batch = ceil(st/batch_size);
            if mod(n_batch, 10) == 0
                fprintf('epoch = %d, batch %d\n', j, n_batch);
            end
            st = st + batch_size;
       end
       model = decode_parameters_to_model(x, model);
       model_list{j} = model;
       test_LL_ELBO(j) = test_ELBO_v2(model, test_data, test_indices);
       test_LL_approx(j) = test_ll_approx_v2(model, test_data, test_indices); 
       fprintf('epoch %d, LL = %g, tau = %g\n', j, test_LL_approx(j), model.tau);
       cur = [];
       cur.models = model_list;
       cur.ELBO = test_LL_ELBO;
       cur.LL = test_LL_approx;
       save(file_name, 'cur');
    end
    %model = decode_parameters_to_model(x, model);
end