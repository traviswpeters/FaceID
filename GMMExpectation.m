function scores = GMMExpectation(X, k, means, covariances, mcoefficients)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assign each point in X a score for each Gaussian (cluster), k.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% > Note: a possible error with computing covariances is fixed by adding this
% small value and recomputing the cov. 
covDataFix = 0.0000001;

% Allocate space for scoring (each pt. has a 'score' representing how likely it
% is that it belongs to one of the k clusters). 
scores = zeros(size(X,1), k); 

% For each data point...
for i=1:size(X,1)
    pt = X(i,:);
    
    % For each cluster...
    sum_y = 0;
    for c=1:k

        % Compute the normal distribution for point i with cluster c data
        % > Note: we could just add the covDataFix first and never wait for 
        %   the error to crop up - if we are modifying the covariance for each
        %   cluster than every data pt. is treated equally...
        try
            y = mvnpdf(pt, means(c,:), covariances{c,:});
        catch error
           covariances{c,:} = covariances{c,:} + covDataFix;
           y = mvnpdf(pt, means(c,:), covariances{c,:});
        end       
        
        % Multiply distribution value by the mixture coefficient.
        ym = mixtureCoeffs(c,1) * y;

        % Update the sum
        sum_y = sum_y + ym; 
        
        % Compute the total cluster's assignment score for the current point
        scores(i, c) = ym; 
        
    end
    
    % Normalize each score by dividing through by sum_y
    scores(i,:) = scores(i,:) / sum_y;
    
end

end

