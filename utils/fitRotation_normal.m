function [RAll, objVal, data] = fitRotation_normal(U, data)
nEle = size(data.N,1);

RAll = zeros(3,3,nEle); % all rotation
objVal = 0;

for kk = 1:nEle
    a = data.A(kk);
    E = data.E_f{kk};
    W = data.W_f{kk};
    dV = data.dV_f{kk};
    dU = (U(E(:,2),:) - U(E(:,1),:))';
        
    n = data.N(kk,:)';
    n_desire = data.t(kk,:)';

    % ortho procustes
    S = dV * W * dU' + (data.lambda * a * n * n_desire');
    R = fit_rotation(S);

    % compute energy
    objVal = objVal + trace((R*dV-dU) * W * (R*dV-dU)') ...
        + data.lambda * a * (R*n - n_desire)'*(R*n - n_desire);
    
    RAll(:,:,kk) = R;
end