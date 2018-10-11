clear all;
close all;
rng('default');

addpath_recurse('./util');
addpath_recurse('./tensor_toolbox_2.6');
addpath_recurse('./lightspeed');
addpath_recurse('./minFunc_2012');

%predicting with different models
R = 1; %1,2,5,8
time_step = 5; %10,20,30
load('../data/911-test-hybrid-baselines.mat')
load(strcat('cp-markov-pp-911-R-', num2str(R), '-TimeStep-', num2str(time_step), '-models.mat'));
res_file = strcat('cp-markov-pp-911-R-', num2str(R), '-TimeStep-', num2str(time_step), '-res.mat');
n_model = length(models);
n_test = length(data.test);
ll_all = zeros(n_model, 1);
ll = zeros(n_model, 2);
for k=1:n_model
    model = models{k};
    ll_all(k) = predictive_log_likelihood_raw(model, data.all);
    curr = zeros(n_test,1);
    for j=1:n_test
        curr(j) = predictive_log_likelihood_raw(model, data.test{j});
    end
    ll(k,:) = [mean(curr), std(curr)/sqrt(n_test)];
end
res = [];
res.all = ll_all;
res.test = ll;
%save(res_file, 'res');

max(ll_all)
[v,i] = max(ll(:,1));
ll(i,:)
