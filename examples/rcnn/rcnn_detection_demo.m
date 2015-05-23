function info = rcnn_detection_demo(varargin)

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

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
