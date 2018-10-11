%calc distance between one entry and another set of entries
%dist(x,y) = exp(-frac{1}{2\lam}\| \x-\y |^2)
function vals = get_dist_log(U, dim, d, lam, ind, other_indices)
    nmod = length(U);    
    x = zeros(1, d);
    Y = zeros(size(other_indices,1),d);
    st = 0;
    for k=1:nmod
        x(st+1:st+dim(k)) = U{k}(ind(k),:);
        Y(:, st+1:st+dim(k)) = U{k}(other_indices(:,k),:);
        st = st + dim(k);
    end
    X = repmat(x, size(Y,1), 1);
    %not take exp here, return log val
    vals = -sum((X - Y).^2 , 2)/lam;
    
end