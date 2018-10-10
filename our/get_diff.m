%return X - Y 
function res = get_diff(U, dim, d, ind, other_indices)
    nmod = length(U);
    x = zeros(1, d);
    Y = zeros(size(other_indices,1),d);
    st = 0;
    for k=1:nmod
        x(st+1:st+dim(k)) = U{k}(ind(k),:);
        Y(:, st+1:st+dim(k)) = U{k}(other_indices(:,k),:);
        st = st + dim(k);
    end
    res =  repmat(x, size(Y,1), 1) - Y; 
end