function [trans] = dim_reduction(p,image_2d,image_3d)

M = image_2d;

q = size(image_3d);

sigmaZ = hyperCov(M);

dX = zeros(q(1)-1, q(2), q(3));
for i=1:(q(1)-1)
    dX(i, :, :) = image_3d(i, :, :) -image_3d(i+1, :, :);
end
dX = hyperConvert2d(dX);
sigmaN = hyperCov(dX);
[U,S,E] = svds(sigmaN,p-1);
F = E*inv(sqrt(S));

sigmaAdj = F'*sigmaZ*F;
[U,S,G] = svds(sigmaAdj,p-1);
H = F*G;

trans = H.';
end