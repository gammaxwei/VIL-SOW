%% Demo: VILSOW on Emotions Dataset
% =========================================================================
% This script demonstrates the VILSOW feature selection algorithm on the 
% 'emotions' multi-view dataset.
%
% Workflow:
% 1. Load Data (emotions.mat)
% 2. Split Data (80% Train, 20% Test)
% 3. Run VILSOW Feature Selection
% 4. Evaluate using ML-KNN classifier (if available)
% =========================================================================

clear; clc; close all;

%% 1. Configuration & Parameters
% -------------------------------------------------------------------------
% Add paths to necessary functions (ensure these folders exist)
addpath('./functions'); 
addpath('./datasets');

dataset_name = 'emotions';
data_path = fullfile('./datasets', [dataset_name, '.mat']);

% Algorithm Hyperparameters (Example values)
alpha  = 0.1;
beta   = 0.01;
gamma  = 0.01;
lambda = 1.0;

% VILSOW Options
opts.maxIte = 30;
opts.flag   = 1; % Plot convergence curve
opts.saveDir = './results';
opts.datasetName = dataset_name;

%% 2. Load and Preprocess Data
% -------------------------------------------------------------------------
fprintf('Loading dataset: %s...\n', data_path);

if ~exist(data_path, 'file')
    error('Dataset file not found: %s. Please place emotions.mat in ./datasets/', data_path);
end

data = load(data_path);

% Handle variable naming variations
if isfield(data, 'X_view'), X = data.X_view; else, error('Variable X_view not found'); end
if isfield(data, 'label'),  Y = double(data.label);  else, error('Variable label not found'); end

% Check label format (Convert 0/1 to -1/1 for MLKNN if necessary, keeping 0/1 for VILSOW)
% Note: VILSOW usually expects 0/1 or indicator matrix. MLKNN often expects -1/1.
Y_mlknn = Y;
Y_mlknn(Y_mlknn == 0) = -1; % Prepare labels for ML-KNN

% Data Split: 80% Train, 20% Test
n_samples = size(Y, 1);
n_train = floor(0.8 * n_samples);

fprintf('Samples: %d | Train: %d | Test: %d\n', n_samples, n_train, n_samples - n_train);

% Split Labels
Y_train = Y(1:n_train, :);
Y_train_mlknn = Y_mlknn(1:n_train, :);
Y_test_mlknn  = Y_mlknn(n_train+1:end, :);

% Split Features (Multi-view Cell Array)
X_train_cell = cell(1, length(X));
X_concat_train = [];
X_concat_test  = [];

for v = 1:length(X)
    % Normalize view data (Min-Max normalization)
    view_data = double(X{v});
    min_v = min(view_data);
    max_v = max(view_data);
    view_data = (view_data - min_v) ./ (max_v - min_v + eps);
    
    % Split
    X_train_cell{v} = view_data(1:n_train, :);
    
    % Concatenate for Evaluation
    X_concat_train = [X_concat_train, view_data(1:n_train, :)];
    X_concat_test  = [X_concat_test,  view_data(n_train+1:end, :)];
end

[~, n_total_features] = size(X_concat_train);

%% 3. Run Feature Selection (VILSOW)
% -------------------------------------------------------------------------
fprintf('Running VILSOW feature selection...\n');
tic;
[feature_rank, obj_curve] = VILSOW(X_train_cell, Y_train, alpha, beta, gamma, lambda, opts);
runtime = toc;
fprintf('Feature selection finished in %.4f seconds.\n', runtime);

%% 4. Evaluation (ML-KNN)
% -------------------------------------------------------------------------
% Select top 20% features
select_ratio = 0.2;
k_feat = ceil(select_ratio * n_total_features);
selected_idx = feature_rank(1:k_feat);

fprintf('Evaluating Top %d features (%.0f%%)...\n', k_feat, select_ratio*100);

% Check if ML-KNN functions exist
if exist('MLKNN_train', 'file') && exist('MLKNN_test', 'file')
    
    % Prepare data subsets
    X_tr_sub = X_concat_train(:, selected_idx);
    X_te_sub = X_concat_test(:, selected_idx);
    
    % ML-KNN Parameters
    K_neighbor = 10;
    Smooth = 1;
    
    % Train
    [Prior, PriorN, Cond, CondN] = MLKNN_train(X_tr_sub, Y_train_mlknn', K_neighbor, Smooth);
    
    % Test
    [HammingLoss, RankingLoss, Coverage, AvgPrec, MacroF1, MicroF1, OneError] = ...
        MLKNN_test(X_tr_sub, Y_train_mlknn', X_te_sub, Y_test_mlknn', K_neighbor, Prior, PriorN, Cond, CondN);
    
    % Display Results
    fprintf('\n---------------------------------\n');
    fprintf('Evaluation Results (ML-KNN):\n');
    fprintf('---------------------------------\n');
    fprintf('Average Precision : %.4f\n', AvgPrec);
    fprintf('Hamming Loss      : %.4f\n', HammingLoss);
    fprintf('Ranking Loss      : %.4f\n', RankingLoss);
    fprintf('Coverage          : %.4f\n', Coverage);
    fprintf('One Error         : %.4f\n', OneError);
    fprintf('Macro F1          : %.4f\n', MacroF1);
    fprintf('Micro F1          : %.4f\n', MicroF1);
    fprintf('---------------------------------\n');
    
else
    warning('ML-KNN functions (MLKNN_train/MLKNN_test) not found in path.');
    fprintf('Selected Feature Indices:\n');
    disp(selected_idx(1:min(10, end))'); % Show top 10 indices
end

fprintf('Done.\n');