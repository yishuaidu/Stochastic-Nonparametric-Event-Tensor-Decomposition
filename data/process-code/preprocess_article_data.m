clear all;
close all;

%pre-process simluation data
%find parents/children candidates, separate training/testing

load('../article_user.mat');
ind_events = datas.ind;
ind_events = ind_events + 1;
% max_ind = max(ind_events);
% hmap1 = zeros(max_ind(1),1);
% hmap2 = zeros(max_ind(2),1);
% hmap1(unique(ind_events(:,1))) = (1:length(unique(ind_events(:,1))))';
% hmap2(unique(ind_events(:,2))) = (1:length(unique(ind_events(:,2))))';
% ind_events(:,1) = hmap1(ind_events(:,1));
% ind_events(:,2) = hmap2(ind_events(:,2));


T = datas.T;
events = datas.e;
%day = 3600*24;
hour = 3600;
%events = events/day;
%T = T/day;
events = events/hour;
T = T/hour;
delim = 50000;
test_sz = 10000;
Dmax = 1; %maximum one hour
train_events = events(1:delim);
test_events = events(delim+1:delim+test_sz);


tr = [];
tr.e = train_events;
tr.ind = ind_events(1:delim,:);
[par, child] = locate_driving_candidate(tr.e, Dmax);
tr.par = par;
tr.child = child;
data = tr;
data.T = tr.e(end);
save('../article-train.mat', 'data');
max_par = 0;
min_par = data.T;
for j=1:length(par)
    if length(par{j})>max_par
        max_par = length(par{j});
    end
    if length(par{j})<min_par
        min_par = length(par{j});
    end
end
fprintf('max par = %d\n', max_par);
fprintf('min par = %d\n', min_par);


te = [];
te.e = test_events;
te.ind = ind_events(delim+1:delim+test_sz,:);

[par, child] = locate_driving_candidate(te.e, Dmax);
te.par = par;
te.child = child;
data = te;
data.T1 = te.e(1);
data.T2 = te.e(end);
save('../article-test.mat', 'data');

