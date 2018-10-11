addpath_recurse('../../RFP-PP/util');
addpath_recurse('../../RFP-PP/tensor_toolbox_2.6');
addpath_recurse('../../RFP-PP/lightspeed');

n_test = 5;
train = load('../ufo-train-hybrid-1-300.mat');
nvec = max(train.data.ind);
%test_data = '../ufo-test-hybrid-1-300-all-remaining.mat';
test_data = '../ufo-test-hybrid-1-300.mat';
load(test_data);
res_file = '../ufo-test-hybrid-baselines-10K.mat';
%res_file = '../ufo-test-hybrid-baselines.mat';
all = [];
all.e_count = sptensor(data.ind(1,:), 1, nvec);
for n=2:size(data.ind,1)
    sub = data.ind(n,:);
    all.e_count(sub) = all.e_count(sub) + 1;
end
all.T = max(data.e)-min(data.e);
all.test_ind = find(all.e_count);
all.test_vals = all.e_count(all.test_ind);
all.test_T = all.T;


ind_all = find(all.e_count);
test = cell(n_test, 1);
seg_len = ceil(length(data.e)/n_test);
for k=1:n_test
    test{k}.e_count = sptensor(ind_all, ones(size(ind_all,1),1), nvec);
    st = 1 + (k-1)*seg_len;
    ed = min(k*seg_len, length(data.e));
    for n = st:ed
        sub = data.ind(n,:);
        test{k}.e_count(sub) = test{k}.e_count(sub) + 1;
    end
    
    test{k}.test_ind = find(test{k}.e_count);
    test{k}.test_vals = test{k}.e_count(test{k}.test_ind) - 1;
    test{k}.test_T = data.e(ed) - data.e(st);
end

data = [];
data.all = all;
data.test = test;
save(res_file,'data');
res_file
    