%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Authors       : Haider Syed & Travis Peters
% Affiliation   : Dartmouth College
% Last Modified : October 2013
%
% This implementation based on the OpenCV algorithmic description for EigenFaces:
%
% http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html#face-recognition
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function eigenFaces(probeImgName, trainfnames)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Inputs: 
%
% probeImgName - File name of the test image.
%                Ex: 'test.jpg'
%
% trainfnames  - Cell structure containing the file names of the training images.
%                Ex: {'t1.jpg', 't2.jpg', 't3.jpg'}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Probe image
imProbe = imread(probeImgName);
[m,n,~] = size(imProbe);                %record size of original probe image
imProbe = im2double(rgb2gray(imProbe)); %convert image to grayscale and double
imProbe = imProbe(:);                   %change the image into a row-vector

%Read in all training images.  
%  imSet = [N x T matrix]:
%    N - length of image vector
%    T - num of Images
imTraining = readImages(trainfnames);

%Get a mean training image from the training images by computing the mean of
%each pixel along the rows (recall, imTraining is a list of length(trainfnames) 
%images and each image is fully represented in a single column of imTraining). 
meanTraining = mean(imTraining')';

%Subtract mean of training images from training images
diffTraining = imTraining - repmat(meanTraining, [1 size(imTraining, 2)]);

%Compute the covariance
covariance = diffTraining' * diffTraining;

%Compute eigenvalues/eigenvectors of the covariance matrix
%NOTE: U contains the eigenvectors.
[V, eigValues]=eig(covariance);
U = diffTraining*V;

%Normalize the eigenvectors
for i = 1:size(U,2)
   U(:,i) = U(:,i)/norm(U(:,i));
end

%Compute the weight vector for the probe image
weightsProbe = U'*(imProbe - meanTraining);

%Compute the weight vectors for each of the training images
for i = 1:size(diffTraining,2)
    weightsTrain(:,i) = U'*(imTraining(:,i) - meanTraining);
end

%Now, compute the differences between the probe image and each of the
%training images
weightDiff = weightsTrain - repmat(weightsProbe,[1 size(weightsTrain, 2)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Use ONE of the following metrics to determine the value e_rec.
%The minimum e_rec value corresponds to the best match we've found.

%(1) Euclidean distance used as metric to find e_rec
e_rec = sqrt(sum(weightDiff.^2));

%(2) Mahalanobis distance used as metric to find e_rec
%e_rec=zeros(1,length(eigVal));
%for i = 1:length(eigVal)
%    e_rec(:,i) = sqrt(sum( (weightDiff(:,i).^2)/eigVal(i) ));
%end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ??? Do we need this...?
eigVal=diag(eigValues);

%Determine the minimum e_rec value & its position in the training set
e_rec
[~, i] = min(e_rec);

%Display the probe image and the best match (if found)
figure
subplot(121), imshow(reshape(imProbe,[m,n]))
subplot(122), imshow(reshape(imTraining(:,i),[m,n]))


function imSet = readImages(fname)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Inputs: 
%
% fname - Cell structure containing training image names
%
%Outputs:
%
% imSet - [N x T matrix], T - num of Images, N - length of image vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imSet=[];
for i = 1:length(fname)
    oneImg = im2double(rgb2gray(imread(fname{i}))); %Convert images to grayscale and double
    imSet = [imSet oneImg(:)];                      %Add each image as a column in imSet matrix
end
