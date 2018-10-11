clear all;
close all;
rng('default');

addpath_recurse('./util');
addpath_recurse('./tensor_toolbox_2.6');
addpath_recurse('./lightspeed');
addpath_recurse('./minFunc_2012');

R = 1; %1,2,5,8
test_data = '../data/911-test-hybrid-baselines-10k.mat';
load(test_data);
load(strcat('rfp-pp-911-R-', num2str(R), '-models.mat'));
res_file = strcat('rfp-pp-911-R-', num2str(R),'-res.mat');
n_model = length(models);
n_test = length(data.test);
ll_all = zeros(n_model, 1);
elbo_all = zeros(n_model, 1);
ll = zeros(n_model, 2);
elbo = zeros(n_model, 2);
for k=1:n_model
    model = models{k};
    ll_all(k) = predictive_log_likelihood_approx(model, data.all);
    elbo_all(k) = predictive_log_ELBO(model, data.all);
    curr = zeros(n_test,1);
    curr_e = zeros(n_test,1);
    for j=1:n_test
        curr(j) = predictive_log_likelihood_approx(model, data.test{j});
        curr_e(j) = predictive_log_ELBO(model, data.test{j});
    end
    ll(k,:) = [mean(curr), std(curr)/sqrt(n_test)];
    elbo(k,:) = [mean(curr_e), std(curr_e)/sqrt(n_test)];
end
res = [];
res.all = ll_all;
res.elbo_all = elbo_all;
res.test = ll;
res.test_elbo = elbo;
save(res_file, 'res');
max(ll_all)
[v,i] = max(ll(:,1));
ll(i,:)

