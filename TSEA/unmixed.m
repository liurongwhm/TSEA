function [rmse]=unmixed(mixed_pixel,endmember,unmixed_method)
% Unmix image with the extracted endmembers by least square method
%Input£∫
%   mixed_pixel: Mixed pixel matrix
%   endmember: Endmember vectors
%   unmixed_method: 
%       1: Unconstrained least squares (Ucls)
%       2: Sum-to-one constrained least squares (Scls)
%       3: Nonnegative constrained least squares (Ncls)
%       4: Fully constrained least squares (Fcls)
%Output£∫
%   abundance: Abundance of each endmember
%   re_mixed: Reconstructed image matrix
%   residual_error: Residuals matrix

[~,pixel_n] = size(mixed_pixel);
[band_n,endmember_n] = size(endmember);

switch unmixed_method
    case 1
%         abundance = pinv(endmember)*mixed_pixel;
        abundance = (endmember' * endmember) \ endmember' * mixed_pixel;
        abundance = max(0,abundance);
        abundance = abundance ./ (ones(endmember_n,1) * sum(abundance));
    case 2
        endmember_1 = ones(band_n+1,endmember_n);
        endmember_1(1:band_n,:) = endmember;
        mixed_pixel_1 = ones(band_n+1,pixel_n);
        mixed_pixel_1(1:band_n,:) = mixed_pixel;
        abundance = pinv(endmember_1)*mixed_pixel_1;
    case 3
        abundance = zeros(endmember_n,pixel_n);
        for i=1:pixel_n
            abundance(:,i) = lsqnonneg(endmember,mixed_pixel(:,i));
        end
    case 4
        endmember_1 = ones(band_n+1,endmember_n);
        endmember_1(1:band_n,:) = endmember;
        mixed_pixel_1 = ones(band_n+1,pixel_n);
        mixed_pixel_1(1:band_n,:) = mixed_pixel;
        abundance = zeros(endmember_n,pixel_n);
        for i=1:pixel_n
            [abundance(:,i) ,~,~,EXITFLAG] = lsqnonneg(endmember_1,mixed_pixel_1(:,i));
            if EXITFLAG == 0
                disp(num2str(i));
            end
        end
    otherwise
        disp(' ‰»Î¥ÌŒÛ£°');
end
% abundance(abs(abundance)<10^(-4))=0;

re_mixed = endmember*abundance;
residual_error = mixed_pixel-re_mixed;
rmse = sum(sqrt(sum(residual_error.^2)/band_n))/pixel_n;
