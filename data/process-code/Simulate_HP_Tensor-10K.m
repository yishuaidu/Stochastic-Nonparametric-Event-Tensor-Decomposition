
clear all;
close all;
addpath_recurse('./Hawkes-Process-Toolkit-master');
addpath_recurse('./tensor_toolbox_2.6');
rng('default');
%generate tensor first
p = [1/2, 1/2];
centers = [1,1; -1,-1];
v = 0.01;
n = 10;
nmod = 3;
U = cell(nmod, 1);
label = cell(nmod, 1);
for k=1:nmod
    U{k} = zeros(n,2);
    label{k} = ones(n,1);
    U{k}(1:n/2,:) = mvnrnd(centers(1,:), v*eye(2), n/2);
    U{k}(n/2+1:end,:) = mvnrnd(centers(2,:), v*eye(2), n/2);
    label{k}(n/2+1:end) = 2;
end
input = zeros(n*n*n, 6);
ind = zeros(n*n*n,3);
st = 1;
for i=1:n
    for j=1:n
        for k=1:n
            input(st,:) = [U{1}(i,:), U{2}(j,:), U{3}(k,:)];
            ind(st,:) = [i,j,k];
            st = st + 1;
        end
    end
end
f1 = input(:,1:2);
f2 = input(:, 3:4);
f3 = input(:, 5:6);
f = sum(abs(f1-f2).^2,2) + sum(abs(f1-f3).^2,2) + sum(abs(f2-f3).^2,2);
f = sqrt(f);
t = tenzeros([n,n,n]);
t(ind) = f(:);
%f = tenfun(@(x) 0.2*log(x.^2+x+1) - cos(x), t);
f = tenfun(@(x) -log(x.^2+x+1) - cos(x), t);
P = sum(input.*input,2);
N = size(input,1);
Dist = repmat(P',N,1) + repmat(P,1,N) - 2*(input*input');
lam = 0.02;
Dist = exp(-Dist/lam);


%options.N = 50; % the number of sequences
options.N = 1;
options.Nmax = 10000; % the maximum number of events per sequence
options.Tmax = 100; % the maximum size of time window
options.tstep = 0.2;% the step length for computing sup intensity
options.M = 50; % the number of steps
options.GenerationNum = 5; % the number of generations
D = N; % the dimension of Hawkes processes


para1.mu = exp(f(ind));
para1.A = Dist;
para1.A = reshape(para1.A, [D, 1, D]);
para1.w = 1;

para2 = para1;
para2.kernel = 'myexp';
para2.bta = 0.1;
para2.tau = 10;
para2.DMax = 0.01;
para2.landmark = 0;
Seq = Simulation_Thinning_HP(para2, options);
Seq.para = para2;
Seq.ind = ind;
save('Simu.mat', 'Seq');



