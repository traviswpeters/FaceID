% An input image (attempt to recognize).
probeImgName = 'probe.jpg';

% A set of images to compare probe image against.
trainfnames = {'tr1.jpg', 'tr2.jpg', 'tr3.jpg','tr4.jpg','tr5.jpg'};

% Compare probe image vs. training images by way of computing EigenFaces.
eigenFaces(probeImgName, trainfnames)