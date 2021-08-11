clc; clear all; close all;
addpath('./utils/')

% load input mesh
[V,F] = readOBJ('./input_shapes/spot.obj');
nV = size(V,1); % number of vertices

% load style mesh (should be simple premitives)
[Vs,Fs] = readOBJ('./style_shapes/cone.obj');
Ns = normalizerow(normals(Vs,Fs));
Ns = unique(Ns, 'rows');
tree_Ns = KDTreeSearcher(Ns);

% precomputation
lambda = 1;
data = precomputation(V,F,lambda);

% compute desired normals (t) for each vertex
N = per_vertex_normals(V,F); 
idx = knnsearch(tree_Ns, N, 'K', 1);
data.t = Ns(idx,:);

%% start optimization
U = V; % output vertex positions

% optimization parameter
tolerance = 1e-4; 
maxIter = 100;

% we have to pin down at least one vertex to avoid the mesh flies away
b = F(1,1); 
bc = U(b,:);

objHis = [];
UHis = zeros(size(V,1), size(V,2), maxIter+1);
UHis(:,:,1) = U;

for iter = 1:maxIter
    
    % local step
    [RAll, objVal, data] = fitRotation_normal(U, data);
    
    % save optimization info
    objHis = [objHis objVal];
    UHis(:,:,iter+1) = U; 
    
    % global step
    Rcol = reshape(permute(RAll,[2,1,3]),1,nV*3*3);
    RHScol = data.K' * Rcol';
    RHS = reshape(RHScol,size(RHScol,1)/3, 3);
    UPre = U;
    [U,data.preF] = min_quad_with_fixed(data.LHS,RHS,b,bc,[],[],data.preF);
    
    % plot
    if mod(iter-1,5) == 0
        figure(1)
        subplot(1,2,1)
        tsurf(F,V);
        axis equal
        subplot(1,2,2)
        tsurf(F,U);
        axis equal
        drawnow
    end
    
    % (optional) update "data.t" using the deformed mesh {U,F}
    N = per_vertex_normals(U,F); 
    idx = knnsearch(tree_Ns, N, 'K', 1);
    data.t = Ns(idx,:);

    % check whether to stop optimization
    dU = sqrt(sum((U - UPre).^2,2));
    dUV = sqrt(sum((U - V).^2,2));
    reldV = max(dU) / max(dUV);
    fprintf('iter: %d, obj: %d, reldV: %d\n', ...
        [iter, objVal, reldV]);
    
end
writeOBJ('outupt.obj', U, F)


