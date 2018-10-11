clear all;
close all;

%pre-process simluation data
%find parents/children candidates, separate training/testing

load('../ufo.mat');
ind_events = datas.ind;
ind_events = ind_events + 1;
max_ind = max(ind_events);
hmap1 = zeros(max_ind(1),1);
hmap2 = zeros(max_ind(2),1);
hmap1(unique(ind_events(:,1))) = (1:length(unique(ind_events(:,1))))';
hmap2(unique(ind_events(:,2))) = (1:length(unique(ind_events(:,2))))';
ind_events(:,1) = hmap1(ind_events(:,1));
ind_events(:,2) = hmap2(ind_events(:,2));


T = datas.T;
events = datas.e;
day = 3600*24;
events = events/day;
T = T/day;
delim = 40000;
test_sz = 10000;
Dmax = 3; %maximuim 3 days
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
save('../ufo-train.mat', 'data');

max_id = size(unique(tr.ind(:,2)));

te = [];
te.e = test_events;
te.ind = ind_events(delim+1:delim+test_sz,:);
sel = (te.ind(:,2)<=max_id);
te.e = te.e(sel);
te.ind = te.ind(sel,:);

[par, child] = locate_driving_candidate(te.e, Dmax);
te.par = par;
te.child = child;
data = te;
data.T1 = te.e(1);
data.T2 = te.e(end);
save('../ufo-test.mat', 'data');

