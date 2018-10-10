%extract the parameters from the vector x and stored them in model
function model = decode_parameters_to_model(x, model)    
    [U, Um, mu, L, ker_param, a, b, tau] = decode_parameters(x, model);
    model.U = U;
    model.Um = Um;
    model.mu = mu;
    model.L = L;
    model.ker_param = ker_param;
    model.a = a;
    model.b = b;
    model.tau = tau;
end