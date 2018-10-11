clear all;
close all;
rng('default');

addpath_recurse('./util');
addpath_recurse('./tensor_toolbox_2.6');
addpath_recurse('./lightspeed');
addpath_recurse('./minFunc_2012');

train_file = '../data/ufo-train-hybrid-1-300.mat';
test_file = '../data/ufo-test-hybrid-1-300.mat';
R = 1; %1,2,5,8
res_file = strcat('./rfp-pp-ufo-R-', num2str(R),'-all.mat');
model_file = strcat('./rfp-pp-ufo-R-',num2str(R), '-models.mat');
decay = 0.95;
nepoch = 100;




load(train_file);
nvec = max(data.ind);
nmod = size(nvec,2);
data.e_count = sptensor(data.ind(1,:), 1, nvec);
for n=2:size(data.ind,1)
    sub = data.ind(n,:);
    data.e_count(sub) = data.e_count(sub) + 1;
end
data.T = max(data.e);

data.train_subs = find(data.e_count);
data.y_subs = data.e_count(data.train_subs);
data.tensor_sz = nvec;

test = load(test_file);
test = test.data;
test.e_count = sptensor(test.ind(1,:), 1, nvec);
for n=2:size(test.ind,1)
    sub = test.ind(n,:);
    test.e_count(sub) = test.e_count(sub) + 1;
end
test.T = max(test.e)-min(test.e);
data.test_ind = find(test.e_count);
data.test_vals = test.e_count(data.test_ind);
data.test_T = test.T;

%GP tensor poission process regression
%parameters setting
model = [];
%model.nmod = nmod;
%the length of latent features
%cp_pp_model = load('../CP-PP/cp-pp-model.mat');
model.R = R;
for k=1:nmod
    model.U{k} = rand(nvec(k), model.R);
    %model.U{k} = 0.1*randn(nvec(k), model.R);
    %model.U{k} = cp_pp_model.model.U{k};
end
model.dim = model.R*ones(nmod,1);
%no. of pseudo inputs
model.np = 100; 
%decay rate (w.r.t. AdDelta)
model.decay = decay;
%no. of epoch
model.epoch = nepoch;
%batch of events
model.batch_size = 100;
%training time period
model.T = data.T; 
%training
[models,test_ll_app, test_elbo] = GPTensorPP_online_advanced(data, model);
save(model_file, 'models');
res = [];
res.epoch = 1:model.epoch;
res.ELBO = test_elbo;
res.LL = test_ll_app;
save(res_file, 'res');
res_file
model_file

