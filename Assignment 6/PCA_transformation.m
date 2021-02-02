function [A,Y,d] = PCA_transformation(x,N)

%calculating the convolution matrix
cov_x = cov(x);

%obtaining the eigen values and vectors
[V,D] = eig(cov_x);

%rearragin the vectors based on the sorted values
d = sort(diag(D),'descend');
[c, ind]=sort(diag(D),'descend'); 
v=V(:,ind);

%selecting the top N features
d = d(1:N);
A = v(:,1:N);

%calulating transform matrix
Y = x*A;

end

