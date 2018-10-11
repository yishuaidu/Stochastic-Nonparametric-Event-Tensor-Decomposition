clear all;
close all;

%pre-process 911 data, using nearest neighbor strategyf
%find parents/children candidates, separate training/testing
%
load('../ufo_startID_with_1.mat');
ind_events = datas.ind;

% max_ind = max(ind_events);
% hmap1 = zeros(max_ind(1),1);
% hmap2 = zeros(max_ind(2),1);
% hmap1(unique(ind_events(:,1))) = (1:length(unique(ind_events(:,1))))';
% hmap2(unique(ind_events(:,2))) = (1:length(unique(ind_events(:,2))))';
% ind_events(:,1) = hmap1(ind_events(:,1));
% ind_events(:,2) = hmap2(ind_events(:,2));





T = datas.T;
events = datas.e;
events = events/(24*3600); %ufo events/24
T = T/(24*3600);  %ufo T/24
delim = 40000;
test_sz = 10000;
Kmax = 300;
Dmax = 5;% 1 ,3,5
train_events = events(1:delim);
tr_inds = ind_events(1:delim,:);
test_events = events(delim+1:delim+test_sz);
te_inds = ind_events(delim+1:delim+test_sz,:);

max_ind = max(ind_events);
hmap1 = zeros(max_ind(1),1);
hmap2 = zeros(max_ind(2),1);
hmap1(unique(tr_inds(:,1))) = (1:length(unique(tr_inds(:,1))))';
hmap2(unique(tr_inds(:,2))) = (1:length(unique(tr_inds(:,2))))';
%ind_events(:,1) = hmap1(tr_inds(:,1));
%ind_events(:,2) = hmap2(ind_events(:,2));
tr_inds(:,1) = hmap1(tr_inds(:,1));
tr_inds(:,2) = hmap2(tr_inds(:,2));


tr = [];
tr.e = train_events;
%tr.ind = ind_events(1:delim,:);
tr.ind = tr_inds;

[par, child] = locate_driving_candidate_hybrid(tr.e, Dmax, Kmax);
tr.par = par;
tr.child = child;
data = tr;
data.T = tr.e(end);
data.Dmax = Dmax;
data.Kmax = Kmax;
file_name_tr = strcat('../ufo-train-hybrid-', num2str(Dmax),'-',num2str(Kmax),'.mat');
save(file_name_tr, 'data');


te_inds(:,1) = hmap1(te_inds(:,1));
te_inds(:,2) = hmap2(te_inds(:,2));

te = [];
te.e = test_events;
%te.ind = ind_events(delim+1:delim+test_sz,:);
te.ind = te_inds;
sel = (te.ind(:,1)~=0);
te.e = te.e(sel);
te.ind = te.ind(sel,:);
sel = (te.ind(:,2)~=0);
te.e = te.e(sel);
te.ind = te.ind(sel,:);



[par, child] = locate_driving_candidate_hybrid(te.e, Dmax, Kmax);
te.par = par;
te.child = child;
data = te;
data.T1 = te.e(1);
data.T2 = te.e(end);
file_name_te = strcat('../ufo-test-hybrid-', num2str(Dmax),'-',num2str(Kmax),'.mat');
save(file_name_te, 'data');
fprintf('train: max # parents = %d\n', max_par_size(file_name_tr));
fprintf('test: max # parents = %d\n', max_par_size(file_name_te));


