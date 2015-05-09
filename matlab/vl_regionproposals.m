function [ rects ] = vl_regionproposals( X )
%VL_REGIONPROPOSALS Candidate object locations using selective search

if isa(X, 'uint8')
    im = im2single(X);
else
    im = single(X);
end

% The params used for "fast mode"
% Colour space conversion functions
colfns = {@rgb2hsv, @myIm2Lab};
% Thresholds for initial segmentation
ks = [50 100];
% Similarity measures
% c=colour, t=texture, s=size, f=fill
simMeasures = {'ctsf', 'tsf'};

rects = [];

for c=1:length(colfns)
    fn = colfns{c};
    csIm = fn(im);
    for k=ks
        minSize = k;
        r = vl_selectivesearch(csIm, k, minSize, simMeasures);
        rects = [rects; r];
        %nRects = size(rects, 1);
    end
end

% Filter rect width or height < 20
minW = 20;
w = rects(:,3) - rects(:,1) + 1;
h = rects(:,4) - rects(:,2) + 1;
rects = rects(w >= minW & h >= minW, :);

% Remove duplicates
rects = unique(rects, 'rows', 'first');

end


function o = myIm2Lab( x )
    % Stupid way to get Lab in expected range
    lab = applycform(im2uint8(x), makecform('srgb2lab'));
    o = im2single(lab);
end
