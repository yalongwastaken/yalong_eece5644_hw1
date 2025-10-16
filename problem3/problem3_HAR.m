% Author: Anthony Yalong
% NUID: 002156860

clear all; close all;

%% HUMAN ACTIVITY RECOGNITION

%% LOAD DATA
X_train = load('data/UCI HAR Dataset/train/X_train.txt');
y_train = load('data/UCI HAR Dataset/train/y_train.txt');
X_test = load('data/UCI HAR Dataset/test/X_test.txt');
y_test = load('data/UCI HAR Dataset/test/y_test.txt');

% join test & train
X = [X_train; X_test];
y = [y_train; y_test];

[N, D] = size(X);
classes = unique(y);
K = length(classes);

% activity names
activity_names = {'Walking', 'Walking Upstairs', 'Walking Downstairs', ...
                  'Sitting', 'Standing', 'Laying'};

fprintf('Dataset properties:\n');
fprintf('  - Samples: %d\n', N);
fprintf('  - Features: %d\n', D);
fprintf('  - Classes: %d activities\n\n', K);

%% ANALYZE CLASS DISTRIBUTION

fprintf('Class distribution:\n');
for k = 1:K
    count = sum(y == k);
    fprintf('  %s: %d samples (%.1f%%)\n', ...
            activity_names{k}, count, 100*count/N);
end

%% DIMENSIONALITY ANALYSIS
fprintf('\nDimensionality challenge:\n');
fprintf('  - Features: %d\n', D);
fprintf('  - Covariance parameters per class: %d\n', D*(D+1)/2);
min_samples_per_class = min(histcounts(y, K));
fprintf('  - Min samples per class: %d\n', min_samples_per_class);

%% NORMALIZATION
mu_features = mean(X);
std_features = std(X);
std_features(std_features == 0) = 1;  % Avoid division by zero
X_norm = (X - mu_features) ./ std_features;

%% ESTIMATE GAUSSIAN PARAMETERS

% setup
mu = cell(K, 1);
Sigma = cell(K, 1);
prior = zeros(K, 1);

% regularization
alpha = 0.1;

for k = 1:K
    idx = (y == k);
    Nk = sum(idx);
    
    % prior probability
    prior(k) = Nk / N;
    
    % mean vector
    mu{k} = mean(X_norm(idx, :))';
    
    % covariance matrix
    variances = var(X_norm(idx, :));
    C_sample = diag(variances);
    
    % regularization
    reg_term = alpha * trace(C_sample) / D;
    Sigma{k} = C_sample + reg_term * eye(D);
end

%% CLASSIFICATION

% log-likelihoods
log_likelihoods = zeros(N, K);

for k = 1:K
    diff = X_norm - mu{k}';
    
    variances = diag(Sigma{k});
    mahal_dist = sum(diff.^2 ./ variances', 2);
    log_det = sum(log(variances));

    % log likelihood
    log_likelihoods(:, k) = -0.5 * (D * log(2*pi) + ...
                                     log_det + mahal_dist);
end

% add log priors
log_posteriors = log_likelihoods + log(prior');

% MAP decision
[~, decisions] = max(log_posteriors, [], 2);

%% CONFUSION MATRIX

confusion = zeros(K, K);
for i = 1:K
    for j = 1:K
        count = sum(decisions == i & y == j);
        confusion(i, j) = count / sum(y == j);
    end
end

fprintf('\nConfusion Matrix (Human Activity Recognition):\n');
fprintf('Predicted\\True\n');
for i = 1:K
    fprintf('%15s: ', activity_names{i});
    for j = 1:K
        fprintf('%.3f  ', confusion(i, j));
    end
    fprintf('\n');
end

% overall accuracy
accuracy = sum(decisions == y) / N;
error_rate = 1 - accuracy;
fprintf('\nOverall Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Error Probability: %.4f\n', error_rate);

% class metrics
fprintf('\nPer-class Performance:\n');
for k = 1:K
    precision = confusion(k, k);
    recall = confusion(k, k);
    fprintf('  %15s: Precision=%.2f%%, Recall=%.2f%%\n', ...
            activity_names{k}, precision*100, recall*100);
end

%% IDENTIFY CONFUSED PAIRS
fprintf('\nMost Confused Activity Pairs:\n');
conf_copy = confusion;
for i = 1:K
    conf_copy(i, i) = 0;
end
[max_conf, max_idx] = max(conf_copy(:));
[pred_idx, true_idx] = ind2sub([K, K], max_idx);
fprintf('  %s misclassified as %s: %.1f%%\n', ...
        activity_names{true_idx}, activity_names{pred_idx}, max_conf*100);

%% VISUALIZATION

[coeff, score, latent] = pca(X_norm);
explained_var = cumsum(latent) / sum(latent);
fprintf('Variance explained:\n');
fprintf('  Variance explained by first 2 PCs: %.1f%%\n', explained_var(2)*100);
fprintf('  Variance explained by first 3 PCs: %.1f%%\n', explained_var(3)*100);
fprintf('  90%% variance: %d PCs needed\n', find(explained_var >= 0.9, 1));
fprintf('  95%% variance: %d PCs needed\n', find(explained_var >= 0.95, 1));

% 2D PCA plot
figure(1);
colors = parula(K);
for k = 1:K
    idx = (y == classes(k));
    scatter(score(idx,1), score(idx,2), 20, colors(k,:), 'filled', ...
            'DisplayName', sprintf('Quality %d', classes(k)));
    hold on;
end
xlabel(sprintf('PC1 (%.1f%%)', latent(1)/sum(latent)*100));
ylabel(sprintf('PC2 (%.1f%%)', latent(2)/sum(latent)*100));
title('har Quality - First 2 Principal Components');
legend('Location', 'best');
grid on;

% 3D PCA plot
figure(2);
for k = 1:K
    idx = (y == classes(k));
    scatter3(score(idx,1), score(idx,2), score(idx,3), ...
             20, colors(k,:), 'filled');
    hold on;
end
xlabel('PC1'); 
ylabel('PC2'); 
zlabel('PC3');
title('har Quality - First 3 Principal Components');
view(45, 30);
grid on;

% variance curve
figure(3);
plot(cumsum(latent(1:min(300,length(latent))))/sum(latent), 'b-', 'LineWidth', 2);
hold on;
plot([1, min(300,length(latent))], [0.9, 0.9], 'r--', 'LineWidth', 1);
plot([1, min(300,length(latent))], [0.95, 0.95], 'g--', 'LineWidth', 1);
xlabel('Number of Principal Components');
ylabel('Cumulative Variance Explained');
title('PCA Variance Explained - har Quality Dataset');
legend('Cumulative Variance', '90% threshold', '95% threshold', 'Location', 'southeast');
grid on;