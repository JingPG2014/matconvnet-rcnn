function info = cnn_rcnn_demo(varargin)

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;

RCNN_PATH = '~/Code/rcnn/' ;

rcnn_file = fullfile(RCNN_PATH, 'data/rcnn_models/voc_2007/rcnn_model_finetuned.mat') ;

if ~exist(rcnn_file, 'file')
    error('First run %s/%s to get the R-CNN models', RCNN_PATH, 'data/fetch_models.sh') ;
end

% Load R-CNN model
l = load(rcnn_file) ;
rcnn_model = l.rcnn_model ;

% Convert to matconvnet
[~, def, ~] = fileparts(rcnn_model.cnn.definition_file) ;
[~, bin, ~] = fileparts(rcnn_model.cnn.binary_file) ;
matconvnetmodel = strcat(def, '-', bin, '.mat') ;
if ~exist(matconvnetmodel, 'file')
    cmd = sprintf('python %s/utils/import-caffe.py %s %s %s',...
                   fullfile(fileparts(mfilename('fullpath')),'..'),...
                   fullfile(RCNN_PATH, rcnn_model.cnn.definition_file),...
                   fullfile(RCNN_PATH, rcnn_model.cnn.binary_file),...
                   matconvnetmodel) ;
    system(cmd) ;
end
net = load(matconvnetmodel) ;

im_u8 = imread(fullfile(RCNN_PATH, 'examples/images/000084.jpg')) ;
im = single(im_u8) ;
%imBGR = single(im(:,:,[3 2 1])) ;

regions = vl_regionproposals(im_u8) ;

feats = zeros(size(regions, 1), 4096, 'single') ;

for i=1:size(regions, 1)
    disp(i)
    r = regions(i,:) ;
    
    % Change from [minY minX maxY maxX] to [minX minY maxX maxY]
    bbox = r([2 1 4 3]) ;
    crop = rcnn_crop_and_preprocess(im, bbox(1), bbox(2), bbox(3), bbox(4),...
            rcnn_model.detectors.crop_padding, rcnn_model.cnn.image_mean) ;
    
    % TODO batching
    res = vl_simplenn(net, crop) ;
    f = squeeze(res(end).x) ;
    feats(i,:) = f ;
    
end

% Get SVM scores
scores = bsxfun(@plus, feats*rcnn_model.detectors.W, rcnn_model.detectors.B) ;

detections = [] ;
for i = 1:length(rcnn_model.classes)
  I = find(scores(:, i) > -1) ;
  scored_boxes = [regions(I, :) i*ones(size(I,1),1) scores(I, i)] ;
  detections = [detections; scored_boxes] ;
end

figure ;
image(im_u8) ;

[~, I] = sort(detections(:,end), 'descend') ;

for i=I'
    bb = double(detections(i,1:4)) ;
    score = detections(i,end) ;
    if score < 0, continue, end
    rectangle('Position',[bb(2), bb(1), bb(4)-bb(2), bb(3)-bb(1)],'EdgeColor',[rand rand rand]) ;
    text(bb(2), bb(1), sprintf('%s %0.3f', ...
        cell2mat(rcnn_model.classes(detections(i,5))), score), 'BackgroundColor', [1 1 1]) ;
end

end

function out = rcnn_crop_and_preprocess(im, x1, y1, x2, y2, padding, meanim)
    % Crop as in https://github.com/BVLC/caffe/blob/master/src/caffe/layers/window_data_layer.cpp#L302
    out_size = size(meanim, 1) ;
    
    % scale box such that we have 'padding' pixels on each side
    % after resizing
    context_scale = out_size / (out_size - 2 * padding) ;
    
    % selective search box dims
    h = (y2 - y1 + 1) ;
    w = (x2 - x1 + 1) ;
    
    % centre
    cx = x1 + w/2 ;
    cy = y1 + h/2 ;
    
    % coords of box when expanded with padding
    x1p = round(cx - w/2 * context_scale) ;
    y1p = round(cy - h/2 * context_scale) ;
    x2p = round(cx + w/2 * context_scale) ;
    y2p = round(cy + h/2 * context_scale) ;
    pboxw = length(x1p:x2p) ;
    pboxh = length(y1p:y2p) ;

    % coords of box with padding clipped to image dims
    x1c = max(1, x1p) ;
    y1c = max(1, y1p) ;
    x2c = min(size(im,2), x2p) ;
    y2c = min(size(im,1), y2p) ;
    
    boxim = im(y1c:y2c, x1c:x2c, :) ;
    cropw = round(size(boxim, 2) * out_size/pboxw) ;
    croph = round(size(boxim, 1) * out_size/pboxh) ;
    
    % Offset at which to place clipped box in output image
    xoffs = round((x1c - x1p) * out_size/length(x1p:x2p)) ;
    yoffs = round((y1c - y1p) * out_size/length(y1p:y2p)) ;
    
    % Enforce that the clipped box does not exceed the output dimensions
    % due to rounding
    if cropw + xoffs > out_size
        cropw = out_size - xoffs ;
    end
    if croph + yoffs > out_size
        croph = out_size - yoffs ;
    end
    
    % Girshick turns of antialiasing to be like OpenCV
    boxim = imresize(boxim, [croph cropw], 'bilinear', 'antialiasing', false) ;
    
    % Subtract mean
    boxim = boxim - meanim(yoffs+1:yoffs+croph, xoffs+1:xoffs+cropw, :) ;
    
    out = zeros(size(meanim), 'single') ;

    out(yoffs+1:yoffs+croph, xoffs+1:xoffs+cropw, :) = boxim ;
end
