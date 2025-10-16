% Author: Anthony Yalong
% NUID: 002156860

clear all; close all;

%% WINE QUALITY RED

%% LOAD DATA
warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
data = readtable('data/wine+quality/winequality-red.csv', 'Delimiter', ';');
warning('on', 'MATLAB:table:ModifiedAndSavedVarnames');

X = table2array(data(:, 1:11)); % features
y = table2array(data(:, 12));   % label

[N, D] = size(X);
classes = unique(y);
K = length(classes);

fprintf('Dataset properties:\n');
fprintf('  - Samples: %d\n', N);
fprintf('  - Features: %d\n', D);
fprintf('  - Classes: %d (scores ', K);
fprintf('%d ', classes');
fprintf(')\n\n');

%% ANALYZE CLASS DISTRIBUTION

fprintf('Class distribution:\n');
for k = 1:K
    class = classes(k);
    count = sum(y == class);
    fprintf('  Quality %d: %d samples (%.1f%%)\n', ...
            class, count, 100*count/N);
end

%% NORMALIZATION
mu_features = mean(X);
std_features = std(X);
X_norm = (X - mu_features) ./ std_features;

fprintf('Feature statistics (before/after normalization):\n');
fprintf('  Feature 1 - Before: mean=%.2f, std=%.2f\n', ...
        mean(X(:,1)), std(X(:,1)));
fprintf('  Feature 1 - After:  mean=%.2e, std=%.2f\n\n', ...
        mean(X_norm(:,1)), std(X_norm(:,1)));

%% ESTIMATE GAUSSIAN PARAMETERS

% setup
mu = cell(K, 1);
Sigma = cell(K, 1);
prior = zeros(K, 1);

% regularization parameter
alpha = 0.01;

for k = 1:K
    class = classes(k);
    idx = (y == class);
    Nk = sum(idx);
    
    % prior probability
    prior(k) = Nk / N;
    
    % mean vector
    mu{k} = mean(X_norm(idx, :))';
    
    % covariance matrix
    C_sample = cov(X_norm(idx, :));
    
    % regularization
    reg_term = alpha * trace(C_sample) / D;
    Sigma{k} = C_sample + reg_term * eye(D);
end

%% CLASSIFICATION

% log-likelihoods
log_likelihoods = zeros(N, K);

for k = 1:K
    diff = X_norm - mu{k}';  % N x D
    inv_Sigma = inv(Sigma{k});
    
    % mahalanobis distance
    mahal_dist = sum((diff * inv_Sigma) .* diff, 2);
    
    % log likelihood
    log_likelihoods(:, k) = -0.5 * (D * log(2*pi) + ...
                                     log(det(Sigma{k})) + ...
                                     mahal_dist);
end

% add log priors
log_posteriors = log_likelihoods + log(prior');

% MAP decision
[~, decisions] = max(log_posteriors, [], 2);
predicted_classes = classes(decisions);

%% CONFUSION MATRIX

confusion = zeros(K, K);
for i = 1:K
    for j = 1:K
        true_class = classes(j);
        pred_class = classes(i);
        count = sum(y == true_class & predicted_classes == pred_class);
        confusion(i, j) = count / sum(y == true_class);
    end
end

fprintf('\nConfusion Matrix (Wine Quality):\n');
fprintf('Pred\\True ');
for k = 1:K
    fprintf('Q%d     ', classes(k));
end
fprintf('\n');

for i = 1:K
    fprintf('Q%d       ', classes(i));
    for j = 1:K
        fprintf('%.3f  ', confusion(i, j));
    end
    fprintf('\n');
end

% overall accuracy
accuracy = sum(predicted_classes == y) / N;
fprintf('\nOverall Accuracy: %.2f%%\n', accuracy * 100);

% class metrics
fprintf('\nPer-class Performance:\n');
for k = 1:K
    precision = confusion(k, k);  % true positives
    recall = confusion(k, k);     % true positives
    fprintf('  Quality %d: Precision=%.2f%%, Recall=%.2f%%\n', ...
            classes(k), precision*100, recall*100);
end

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
title('Wine Quality - First 2 Principal Components');
legend('Location', 'best');
grid on;

% 3D PCA plot
figure();
for k = 1:K
    idx = (y == classes(k));
    scatter3(score(idx,1), score(idx,2), score(idx,3), ...
             20, colors(k,:), 'filled');
    hold on;
end
xlabel('PC1'); 
ylabel('PC2'); 
zlabel('PC3');
title('Wine Quality - First 3 Principal Components');
view(45, 30);
grid on;

% variance curve
figure();
plot(cumsum(latent(1:min(50,length(latent))))/sum(latent), 'b-', 'LineWidth', 2);
hold on;
plot([1, min(50,length(latent))], [0.9, 0.9], 'r--', 'LineWidth', 1);
plot([1, min(50,length(latent))], [0.95, 0.95], 'g--', 'LineWidth', 1);
xlabel('Number of Principal Components');
ylabel('Cumulative Variance Explained');
title('PCA Variance Explained - Wine Quality Dataset');
legend('Cumulative Variance', '90% threshold', '95% threshold', 'Location', 'southeast');
grid on;