function [means, covariances, mcoefficients] = GMMEM(X, k, threshold, maxIters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For k different Gaussians there should be k:
% - averages
% - variances
% - mixing coefficients
%
% Inputs:
% - X         : inputs data points (pixel coordinates)
% - k         : number of Gaussians (clusters)
% - threshold : convergence threshold
% - maxIters  : max number of iterations for Expectation-Maximization (EM)
%
% Outputs:
% - means         :
% - covariances   :
% - mcoefficients :
%
% Method adopted from:
%
% http://www.slideshare.net/petitegeek/expectation-maximization-and-gaussian-mixture-models
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize: use K-Means
% > Note:
%   - old_means is a (k-by-2)
%   - old_covariances has k cells of (2-by-2) covariance matrices
%   - old_mcoefficients is a (k-by-1) matrix
[old_means, old_covariances, old_mcoefficients] = EMInit(X, k);

% Container for log-likelihoods 
% > Note: an ll will have 2 values since we are dealing with cartesian
%   coordinates; thus, ll will be stored as an Nx2 matrix, where N is the number
%   of iterations that are run during the EM algorithm. 
%ll = zeros(maxIters,2); <-- we could pre-allocate for speed?
ll = [];

% Threshold for determining if model is converging
% > Note: compare ll(i) and ll(i-1)
threshold = [threshold threshold];

% Variable to track the state of the model & whether it is converging or not. 
converged = 0;

% Expectation-Maximization Loop
iter = 1;
while ~converged && iter <= maxIters

    % Expectation step
    scores = GMMExpectation(X, k, old_means, old_covariances, old_mcoefficients);

    % Maximization step
    [new_means, new_covariances, new_mcoefficients] = GMMMaximization(X, k, scores);%, old_means, old_covariances, old_mcoefficients);
    
    % Update LogLikelihood
    new_ll = LogLikelihood(X, k, new_means, new_covariances, new_mcoefficients);
    ll = [ll; new_ll];
    
    % Compare new model and previous model to determine if model is converging.
    if iter > 1
        lldiff = abs( ll(iter,:) - ll(iter-1,:) );
        converged = max(lldiff < threshold);
    end
    
    % Update means, covariances, & mcoefficients for next iteration.
    old_means = new_means;
    old_covariances = new_covariances;
    old_mcoefficients = new_mcoefficients;
    
    % Update iter count for termination if EM doesn't converge quick enough.
    iter = iter + 1;
end

% Update return values
means = old_means;
covariances = old_covariances;
mcoefficients = old_mcoefficients; 

end

function [means, covs, coeffs] = EMInit(X, k)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EMInit: use K-Means to group data initially
%  - mean_k  = kmeans mean_k
%  - cov_k   = cov(cluster(k))
%  - coeff_k = (# pts in k / total pts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use kmeans to do initial clustering
[IDX, ~] = kmeans(X, k);

% Initialize cluster data (allocate space)
means = zeros(k,2);
covs = cell(k,1);
coeffs = zeros(k,1);

% Compute cluster data for each cluster, k
for i=1:k
    % Get data pts. that belong to cluster i
    rows = find(IDX == i);
    cluster = X(rows,:);
    
    % The mean for cluster i is just the mean of cluster i
    means(i,:) = mean(cluster);
    
    % The cov for cluster i is just the cov of cluster i
    covs{i} = cov(cluster,1);
    
    % The coeffs for cluster i is: (# pts in cluster i)/(total pts)
    coeffs(i,1) = (size(cluster,1))/(size(X,1));
end

end