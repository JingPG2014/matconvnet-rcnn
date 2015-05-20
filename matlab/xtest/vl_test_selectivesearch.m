function vl_test_selectivesearch
% VL_TEST_SELECTIVESEARCH
%   Check we get similar results to the reference implementation
%   Used reference demo.m with peppers.png, minSize = k = 200
%   HSV colourspace and similarity SSSimColourTextureSizeFillOrig

% standard matlab image
im = imread('peppers.png');

% silly way to replicate demo
datau8 = im2uint8(rgb2hsv(im2double(im)));
data = single(im2double(datau8));

% Check initial segmentation
% We will have different indices of the blobs,
% but the sizes will be (almost) the same, so make a sorted
% list of sizes
%
% refSizes = []; 
% for i=min(blobIndIm(:)):max(blobIndIm(:))
%     refSizes = [refSizes; sum(sum(blobIndIm == i))];
% end
% [sortedRefSizes, sri] = sort(refSizes);
sortedRefSizes = [213,227,229,235,249,251,254,272,272,274,277,284,288,303,313,316,333,...
                  337,348,354,354,360,364,370,370,380,382,391,398,401,405,409,411,439,456,469,477,484,...
                  502,524,530,540,550,585,608,611,635,636,703,743,764,788,826,885,891,926,968,1041,1110,...
                  1542,1687,1762,1808,2014,2603,2725,2859,2867,2918,3016,3222,3258,3321,3330,3631,3875,...
                  3946,4182,5193,5452,6864,10466,26629,59123]';


[rects, initSeg, hists] = vl_selectivesearch(data);
sizes = [];
topIdx = max(initSeg(:));
for i=0:topIdx
    sizes = [sizes; sum(sum(initSeg == i))];
end
[sortedSizes, si] = sort(sizes);

differences = sum(sortedRefSizes ~= sortedSizes)
% Should only be a small number of inconsistencies
% (maybe due to using float images in range 0-1 rather than 0-255 
% for initial segmentation or ambiguities in sorting)
assert(differences < 15);

% Use the region with size 254 for further checks
% Our index si(7) = 20 (or 19 when 0-based as in initSeg)
% Theirs sri(7) = 64

% Check colour histogram
% refColourDesc = colourHist(64,:)
refColourDesc = [0.0420,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0013,0.0052,0.2848,0,0,0,0,0,0,0,...
                 0,0,0,0.0013,0,0,0.0013,0,0,0.0013,0.0013,0,0.0171,0.0709,0.1732,0.0669,0,0,0,0,0.0013,...
                 0.0013,0.0026,0,0.0039,0.0013,0.0052,0.0236,0.0394,0.0499,0.0459,0.0341,0.0525,0.0249,0.0197,...
                 0.0197,0.0079,0,0,0,0,0,0];

colourDesc = hists{2}(:,20)';

% Should be almost identical
diffs = abs(refColourDesc - colourDesc);
assert(max(diffs) < 0.0001);

% Texture histogram
% The reference implementation seems to actually get this wrong, repeating the histogram
% for the 1st channel three times. We fixed this to get the descriptor below.
refTextureDesc = [0.0254,0.0049,0.0026,0.0015,0.0021,0.0011,0.0008,0.0018,0.0005,0.0008,0.0258,0.0051,0.0030,0.0026,0.0021,...
                  0.0013,0.0010,0.0003,0.0005,0,0.0274,0.0041,0.0030,0.0020,0.0011,0.0011,0.0016,0.0008,0.0005,0,0.0253,...
                  0.0067,0.0028,0.0033,0.0021,0.0007,0.0005,0.0003,0,0,0.0261,0.0046,0.0030,0.0015,0.0016,0.0016,0.0010,...
                  0.0010,0.0013,0,0.0253,0.0056,0.0034,0.0026,0.0018,0.0010,0.0005,0.0005,0.0010,0,0.0274,0.0048,0.0030,...
                  0.0011,0.0011,0.0015,0.0015,0.0007,0.0007,0,0.0269,0.0057,0.0028,0.0021,0.0018,0.0020,0,0.0003,0,0,0.0417,...
                  0,0,0,0,0,0,0,0,0,0.0349,0.0044,0.0020,0.0003,0,0,0,0,0,0,0.0397,0.0018,0.0002,0,0,0,0,0,0,0,0.0417,0,0,0,0,...
                  0,0,0,0,0,0.0351,0.0039,0.0008,0.0018,0,0,0,0,0,0,0.0417,0,0,0,0,0,0,0,0,0,0.0415,0.0002,0,0,0,0,0,0,0,0,...
                  0.0366,0.0038,0.0013,0,0,0,0,0,0,0,0.0412,0.0005,0,0,0,0,0,0,0,0,0.0367,0.0028,0.0015,0.0007,0,0,0,0,0,0,...
                  0.0379,0.0030,0.0008,0,0,0,0,0,0,0,0.0405,0.0011,0,0,0,0,0,0,0,0,0.0359,0.0034,0.0018,0.0005,0,0,0,0,0,0,...
                  0.0413,0.0003,0,0,0,0,0,0,0,0,0.0417,0,0,0,0,0,0,0,0,0,0.0402,0.0015,0,0,0,0,0,0,0,0];

textureDesc = hists{1}(:,20)';
% They seem similar, but quite some variation (due to the gradient computation implementations?).
% Not sure how to compare sensibly.

% This function combines multiple selective search
% proposals using different colourspaces and initial
% segmentation parameters
rs = vl_regionproposals(data);
% just a simple "checksum" to see if something changes
assert(sum(rs(:)) == 1618662);

end
