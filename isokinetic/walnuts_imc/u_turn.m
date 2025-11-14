function result = u_turn(O)
    % U-turn condition for HMC/IMC-style WALNUTS
    %
    % O is a cell array of orbit states:
    %   O{k,1} = theta_k   (d-dimensional vector)
    %   O{k,2} = v_k       (d-dimensional vector: rho or u)

    theta_left  = O{1, 1};
    v_left      = O{1, 2};

    theta_right = O{end, 1};
    v_right     = O{end, 2};

    % Force everything to be column vectors
    theta_left  = theta_left(:);
    theta_right = theta_right(:);
    v_left      = v_left(:);
    v_right     = v_right(:);

    dtheta = theta_right - theta_left;   % also column

    % Now all are d×1, so dot is safe
    dot1 = dot(v_right, dtheta);
    dot2 = dot(v_left,  dtheta);

    % U-turn if either end velocity is pointing back toward the other
    result = (dot1 < 0) || (dot2 < 0);
end