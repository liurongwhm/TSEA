function [V] = volume(A)
% Calculate the simplex volume formed by P endmembers (A)
[~,p] = size(A);
% A1 = trans*A;
A2 = [ones(1,p);A];       
V=abs(det(A2)/factorial(p-1));
% V=abs(det(A2));
end
