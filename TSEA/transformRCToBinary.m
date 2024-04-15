function [binarycode] = transformRCToBinary(row,col,P,x)
%  Convert row-column encoding to binary encoding
binarycode = zeros(1,row*col);
for k = 1:P
    r = x(1,k);
    c = x(1,P+k);
    idx = (c-1)*row+r;
    binarycode(1,idx) = 1;
end
end

