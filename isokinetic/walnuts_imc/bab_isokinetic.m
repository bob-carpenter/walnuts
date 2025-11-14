function [theta, rho, W] = bab_isokinetic(logmu, gradlogmu, theta, rho, h, ell, K0)
% BAB-IMC integrator: ell steps of B_{h/2} - A_h - B_{h/2}
%   State: (theta, rho) with ||rho||^2 = 2 K0
%   W accumulates the log-Jacobian contributions from the B-kicks.

    W = 0;
    for k = 1:ell
        % First B half-kick
        [rho_half, dW] = B_half_isokinetic(gradlogmu, theta, rho, h, K0);
        W     = W + dW;

        % Drift (A-step): theta move with constant rho_half
        theta = theta + h * rho_half;

        % Second B half-kick at new theta
        [rho, dW] = B_half_isokinetic(gradlogmu, theta, rho_half, h, K0);
        W     = W + dW;
    end
end


function [rho_plus, dW] = B_half_isokinetic(gradlogmu, theta, rho, h, K0)
% One B_{h/2} kick in isokinetic IMC, with log-Jacobian increment dW.

    g   = gradlogmu(theta);        % score
    xi  = norm(g);                 % ||g||
    zeta = sqrt(2 * K0);           % ||rho|| target norm
    d   = numel(theta);            % dimension

    if xi == 0
        rho_plus = rho;
        dW       = 0;
        return;
    end

    ghat = g / xi;                 % unit score direction
    c    = (ghat' * rho) / zeta;   % cosine between rho and ghat
    delta = (h / 2) * (xi / zeta); % dimensionless step

    % Scalar normalizer
    Z = cosh(delta) + c * sinh(delta);

    % Updated velocity on isokinetic sphere
    rho_plus = (rho + (sinh(delta) + c * (cosh(delta) - 1)) * zeta * ghat) / Z;

    % Re-project exactly onto the sphere to control roundoff
    rho_plus = zeta * rho_plus / norm(rho_plus);

    % Log-Jacobian contribution (on the tangent sphere)
    dW = (d - 1) * log(Z);
end