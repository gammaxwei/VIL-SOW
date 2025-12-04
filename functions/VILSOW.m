function [feature_slc, obj] = VILSOW(X, Y, alpha, beta, gamma, lambda, options)
% VILSOW: Feature selection for Multi-view data via Structural Learning.
%
% Usage:
%   [feature_slc, obj] = VILSOW(X, Y, alpha, beta, gamma, lambda, options)
%
% Inputs:
%   X       - Cell array of matrices, where X{i} is n_samples x d_features.
%   Y       - Label matrix (n_samples x n_classes).
%   alpha   - Parameter for the projection term.
%   beta    - Parameter for the graph regularization term.
%   gamma   - Parameter for the HSIC term.
%   lambda  - Parameter for L2,1 norm regularization.
%   options - (Optional) Struct containing settings:
%             options.maxIte: Maximum iterations (default: 30)
%             options.flag:   1 to plot/save convergence, 0 otherwise (default: 0)
%             options.saveDir: Directory to save plots (default: './results')
%             options.datasetName: Name for the saved plot file.
%
% Outputs:
%   feature_slc - Ranked indices of selected features.
%   obj         - Objective function value per iteration.
%
% Dependencies:
%   - Laplacian_GK.m
%   - HSIC.m (or local implementation)
%   - L21.m (or local implementation)
%   - standard MATLAB functions (lyap, svd, etc.)
%
% ----------------------------------------------------------------------

    %% 1. Parameter Initialization & Validation
    if nargin < 7, options = struct(); end
    if ~isfield(options, 'maxIte'), options.maxIte = 30; end
    if ~isfield(options, 'flag'), options.flag = 0; end
    if ~isfield(options, 'datasetName'), options.datasetName = 'dataset'; end
    if ~isfield(options, 'saveDir'), options.saveDir = './results'; end
    
    eps_val = 1e-5; % Tolerance for division by zero
    rho = 0.1;      % Augmented Lagrangian parameter
    
    num_view = length(X);
    n = size(X{1}, 1);
    l = size(Y, 2);
    k = l - 1;
    
    % Initialize dimensions array
    d = zeros(1, num_view);
    for i = 1:num_view
        d(i) = size(X{i}, 2);
    end

    %% 2. Graph Construction
    % Calculate Label Correlation Graph Laplacian L = A - S
    % NOTE: Ensure Laplacian_GK is in your path
    [A, S] = Laplacian_GK(Y, k); 
    % L = A - S; % L is not explicitly used in update rules, but parts (A-S) are.

    %% 3. Variable Initialization
    P = cell(1, num_view);
    Q = cell(1, num_view);
    V = cell(1, num_view);
    R = cell(1, num_view);
    
    % Auxiliary matrices for L2,1 norm (IRLS method)
    D1 = cell(1, num_view); 
    D2 = cell(1, num_view);
    
    % Centering matrix H (Pre-calculated to save time)
    H_mat = eye(l) - (1/l) * ones(l, l); 

    for j = 1:num_view
        P{j} = rand(l, d(j)); 
        Q{j} = rand(d(j), l);
        V{j} = rand(n, l);
        R{j} = rand(n, l);
    end

    obj = zeros(options.maxIte, 1); % Pre-allocate objective array
    obji = 1; 
    iter = 0;

    %% 4. Optimization Loop
    while iter < options.maxIte
        iter = iter + 1;
        
        % Update Diagonal Matrices D1, D2 (Reweighting for L2,1 norm)
        for j = 1:num_view
            D1{j} = diag(0.5 ./ sqrt(sum(Q{j}.*Q{j}, 2) + eps_val));
            D2{j} = diag(0.5 ./ sqrt(sum(P{j}'.*P{j}', 2) + eps_val));
        end

        for j = 1:num_view
            % --- Update P{j} (Lyapunov Equation) ---
            % Equation form: C11*P + P*C21 = C31 (depends on lyap definition)
            % MATLAB lyap(A,B,C) solves AX + XB = -C. 
            % Based on derivation, adjust signs accordingly.
            C11 = 2*alpha*(V{j}'*V{j}) + 2*beta*(A - S);
            C21 = 2*(X{j}'*X{j}) + 2*lambda*D2{j};
            C31 = -2*Y'*X{j} - 2*alpha*V{j}'*X{j}; 
            
            % Solve Sylvester/Lyapunov equation
            try
                P{j} = lyap(C11, C21, C31);
            catch
                warning('Lyapunov solver failed at view %d iter %d. Using previous P.', j, iter);
            end

            % --- Update V{j} (SVD / Orthogonal Procrustes) ---
            Zv = alpha*X{j}*P{j}' + X{j}*Q{j} + rho*R{j};
            [Uv, ~, Vv] = svd(Zv, 'econ');
            V{j} = Uv * Vv';

            % --- Update R{j} ---
            R{j} = max(V{j}, 0);

            % --- Update Q{j} (Lyapunov Equation) ---
            U_sum = zeros(l, l);
            for jj = 1:num_view
                if(jj ~= j)
                    % Use pre-calculated H_mat
                    U_sum = U_sum + H_mat * (Q{jj}' * Q{jj}) * H_mat;
                end
            end
            tempU = (U_sum + U_sum'); % Ensure symmetry
            
            AA = 2*(X{j}'*X{j}) + 2*lambda*D1{j};
            BB = gamma * tempU;
            CC = -2 * X{j}' * V{j};
            
            try
                Q{j} = lyap(AA, BB, CC);
            catch
                warning('Lyapunov solver failed for Q at view %d. Using previous Q.', j);
            end
        end

        %% 5. Convergence Check
        current_obj = 0;
        for j = 1:num_view
            % Reconstruction Terms
            term1 = norm(X{j}*P{j}' - Y, 'fro')^2;
            term2 = alpha * norm(X{j} - V{j}*P{j}, 'fro')^2;
            term3 = beta * trace(P{j}' * (A - S) * P{j});
            term4 = rho * norm(R{j} - V{j}, 'fro')^2;
            term5 = norm(X{j}*Q{j} - V{j}, 'fro')^2;
            
            % Regularization
            % Note: Check if L21() function is sum of row norms
            term6 = lambda * (calculate_L21(P{j}') + calculate_L21(Q{j})); 
            
            % HSIC Term
            tempHSIC = 0;
            for jj = 1:num_view
                if(jj ~= j)
                    % Note: Ensure HSIC function is available
                    tempHSIC = tempHSIC + calculate_HSIC(Q{j}, Q{jj}, H_mat); 
                end
            end
            
            current_obj = current_obj + term1 + term2 + term3 + term4 + term5 + term6 + gamma*tempHSIC;
        end
        
        obj(iter) = current_obj;
        
        % Check relative error
        if iter > 1
            cver = abs((obj(iter) - obj(iter-1)) / obj(iter-1));
            if cver < 1e-5 && iter > 2
                fprintf('Converged at iteration %d.\n', iter);
                break; 
            end
        end
        obji = obj(iter);
    end
    
    % Trim objective array if converged early
    obj = obj(1:iter);

    %% 6. Feature Selection Ranking
    M = [];
    for j = 1:num_view
        M = [M; P{j}' + Q{j}];
    end
    
    % Score based on L2 norm of the combined weights
    feature_weights = sum(M.^2, 2);
    [~, feature_slc] = sort(feature_weights, 'descend'); 

    %% 7. Visualization & Saving
    if options.flag == 1
        plot_convergence(obj, options.datasetName, options.saveDir);
    end
end

%% --- Helper Functions (Subroutines) ---

function val = calculate_L21(X)
    % Calculate L2,1 norm: Sum of Euclidean norms of rows
    val = sum(sqrt(sum(X.^2, 2)));
end

function val = calculate_HSIC(Q1, Q2, H)
    % Hilbert-Schmidt Independence Criterion using pre-computed H
    % Standard HSIC(A,B) = trace(A*H*B*H) / (n-1)^2 usually, 
    % but here implementation depends on the paper's definition.
    % Assuming Kernel is linear: K1 = Q1'*Q1
    K1 = Q1' * Q1;
    K2 = Q2' * Q2;
    val = trace(K1 * H * K2 * H);
end

function plot_convergence(obj, datasetName, saveDir)
    % Plotting utility
    
    % Graphics Settings
    fontsize = 14; % Reduced slightly for better standard viewing
    lineWidth = 2.0; 
    markerSize = 6;
    
    % Create invisible figure for saving
    hFig = figure('Color', 'w', 'Visible', 'off'); 
    ax = axes('Parent', hFig, ...
        'FontSize', fontsize, ...
        'FontName', 'Times New Roman', ...
        'LineWidth', 1.2); 
    hold(ax, 'on');
    
    plot(obj, '-o', ...
        'Parent', ax, ...
        'LineWidth', lineWidth, ...
        'MarkerSize', markerSize, ...
        'Color', [0.85, 0.1, 0.1], ...       
        'MarkerFaceColor', [1, 0.6, 0.6], ...
        'MarkerEdgeColor', [0.85, 0.1, 0.1]);
        
    xlabel(ax, 'Iteration number');
    ylabel(ax, 'Objective function value');
    grid(ax, 'on');
    box(ax, 'on');
    
    % Set Scientific Notation for Y-axis if values are large
    ax.YAxis.Exponent = 4; 
    
    % Save
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end
    
    fileName = fullfile(saveDir, [datasetName, '_convergence.eps']);
    try
        print(hFig, fileName, '-depsc', '-r300', '-painters');
        fprintf('Convergence plot saved to: %s\n', fileName);
    catch ME
        % SAFE WARNING: Using a specific ID and formatting the message safely
        warning('VILSOW:SavePlotFailed', 'Failed to save plot. Error: %s', ME.message);
    end
    
    close(hFig);
end