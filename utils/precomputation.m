function data = precomputation(V,F,lambda)
    data.V = V;
    data.F = F;
    data.N = per_vertex_normals(V,F); % input face normal
    data.preF = []; % prefactorization of Q
    data.A = full(diag(massmatrix(V,F))); % face area
    data.totalArea = sum(data.A);
    data.cotanW = cotangent(V,F); % cotangent weight
    data.lambda = lambda;
    nV = size(V,1);
    
    % get a list "adjFList" with size |V| where adjF{ii} outputs the 
    % adjacent face indices of a vertex
    i = (1:size(F,1))';
    j = F;
    VT = sparse([i i i],j,1);
    adjFList = cell(size(VT,2),1);
    indices = 1:size(F,1);
    adjFList = cell(size(VT,2),1);
    for ii = 1:size(VT,2)
      adjFList{ii} = indices(logical(VT(:,ii)));
    end
    
    %% construct neighboring info (Nk)
    for kk = 1:nV
        % Nk
        Nk = adjFList{kk};
        
        E_fk = [F(Nk,1) F(Nk,2); ...
              F(Nk,2) F(Nk,3); ...
              F(Nk,3) F(Nk,1)];
        
        W_fk = [data.cotanW(Nk, 3); ...
              data.cotanW(Nk, 1); ...
              data.cotanW(Nk, 2)];
          
        % save info
        data.E_f{kk} = E_fk;
        data.W_f{kk} = diag(W_fk);
        data.dV_f{kk} = (V(E_fk(:,2),:) - V(E_fk(:,1),:))';
    end
    
    %% precomputation for the global step (Q1 and K1)
    QIJV = zeros(nV*3*4*4,3); % construct a long enough list for Q1
    KIJV = zeros(nV*18*3*4,3); % construct a long enough list for K1
    QIdx = 1;
    KIdx = 1;
    for kk = 1:nV
        E_fk = data.E_f{kk};
        W_fk = diag(data.W_f{kk});
        
        nE = size(E_fk,1);
        
        Qi = [E_fk(:,1); E_fk(:,2); E_fk(:,1); E_fk(:,2)];
        Qj = [E_fk(:,2); E_fk(:,1); E_fk(:,1); E_fk(:,2)];
        Qv = [W_fk; W_fk; -W_fk; -W_fk];        
        QIJV(QIdx:QIdx+4*nE-1,:) = [Qi,Qj,Qv];
        QIdx = QIdx + 4*nE;
        
        Ki = [repmat(1,nE*2,1); repmat(2,nE*2,1); repmat(3,nE*2,1)];
        Kj = repmat([E_fk(:,1); E_fk(:,2)], 3, 1);
        Kv = [W_fk .* V(E_fk(:,1),:)-W_fk .* V(E_fk(:,2),:); W_fk .* V(E_fk(:,2),:)-W_fk .* V(E_fk(:,1),:)];
        Kv = Kv(:);
        KIJV(KIdx:KIdx+6*nE-1, :) = [Ki+9*(kk-1), Kj, Kv];
        KIdx = KIdx + 6*nE;
    end
    QIJV(QIdx:end,:) = [];
    KIJV(KIdx:end,:) = [];
    
    Q = sparse(QIJV(:,1),QIJV(:,2),QIJV(:,3),size(V,1),size(V,1));
    
    data.K = sparse([KIJV(:,1); KIJV(:,1)+3; KIJV(:,1)+6], ...
                [KIJV(:,2); KIJV(:,2)+nV; KIJV(:,2)+nV+nV], ...
                [KIJV(:,3);KIJV(:,3);KIJV(:,3)], ...
                9*nV,3*nV);  
            
    data.LHS = Q /2;
end
