function [ rects ] = vl_regionproposals( X )
%VL_REGIONPROPOSALS Candidate object locations using selective search

if isa(X, 'uint8')
    im = im2single(X);
else
    im = single(X);
end

% TODO try different colour spaces
hsv = rgb2hsv(im);
rects = vl_selectivesearch(hsv);

end