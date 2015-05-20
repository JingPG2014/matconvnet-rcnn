function cnn_rcnn_finetune
% First you must run utils/import-ref-models.sh

VOC_DIR = '~/Code/VOCdevkit' ;
VOC_EDITION = 'VOC2007' ;
VOC_IMAGESET = 'trainval' ;

matfile = sprintf('windows_%s_%s.mat', VOC_EDITION, VOC_IMAGESET) ;

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;

% Load VOC dataset

addpath(fullfile(VOC_DIR, 'VOCcode')) ;
VOCopts = [] ;
VOCinit ; % creates VOCopts
image_ids = textread(sprintf(VOCopts.imgsetpath, VOC_IMAGESET), '%s') ;
    
try
    load(matfile) ;
catch
    windows = cell(length(image_ids), 1) ;
    parfor i = 1:length(image_ids)
        disp(i)
        % Load annotation data
        rec = PASreadrecord(sprintf(VOCopts.annopath, image_ids{i})) ;
        
        % Do selective search
        proposals = vl_regionproposals(imread(fullfile(VOC_DIR, rec.imgname))) ;
        proposals = proposals(:, [2 1 4 3]) ; % convert to PASCAL order
        
        ground_truth = cat(1, rec.objects.bbox) ;
        nground = size(ground_truth, 1) ;
        gt_class_indices = zeros(nground, 1) ;
        
        % Find overlap with ground truth boxes
        ovs = zeros(size(proposals, 1), nground) ;
        for g=1:nground
            class_index = find(strcmp(VOCopts.classes, rec.objects(g).class)) ;
            gt_class_indices(g) = class_index ;
            ov = overlap(proposals, ground_truth(g,:)) ;
            ovs(:, g) = ov ;
        end
        
        % Assign label of max overlap box
        [maxovs,I] = max(ovs, [], 2) ;
        
        proposal_classes = gt_class_indices(I) ;
        proposal_classes(maxovs == 0) = 0 ;
        
        windows{i} = [gt_class_indices ones(nground, 1) ground_truth ;...
                      proposal_classes maxovs           proposals] ;
    end
    
    save(matfile, 'windows') ;
end

imids = [] ;
for i=1:size(windows, 1) 
    imids = [imids ; i * ones(size(windows{i}, 1), 1)] ;
end
windows = [cat(1, windows{:}), imids] ;
% columns: class, overlap, bb1, bb2, bb3, bb4, imageid

% Load net
cafferef = fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'data', 'models', 'imagenet-caffe-ref.mat') ;

net = load(cafferef) ;

% Adapt fc8 to pascal
assert(all(net.layers{end-1}.name == 'fc8')) ;

net.layers{end-1}.weights{1} = 0.01 * randn([1 1 4096 21], 'single') ;
net.layers{end-1}.weights{2} = 0.01 * randn([1 21], 'single') ;
net.layers{end-1}.learningRate = single([10 20]) ;

net.layers{end}.type = 'softmaxloss' ;

batch_size = 128 ;
%expdir = '/data/jdt/exp-voc2007-trainval-v3' ;
expdir = '/tmp/exp-voc2007' ;
gpus = [1] ;

imdb.windows = windows ;
imdb.classes = VOCopts.classes ;
imdb.imgpath = VOCopts.imgpath ;
imdb.image_ids = image_ids ;
imdb.averageImage = net.normalization.averageImage ;

% Give fake train and val sets of an appropriate size, since currently
% our getBatch function just generates a batch on the fly from the
% set of windows
[net, info] = cnn_train(net, imdb, @getBatch, 'train', -1*ones(40*batch_size,1),...
                    'val', -1*ones(1*batch_size,1), 'batchSize', batch_size,...
                    'learningRate', 0.001, 'gpus', gpus, 'expDir', expdir,...
                    'continue', true, 'numEpochs', 900) ;

% Decrease learning rate
[net, info] = cnn_train(net, imdb, @getBatch, 'train', -1*ones(40*batch_size,1),...
                    'val', -1*ones(1*batch_size,1), 'batchSize', batch_size,...
                    'learningRate', 0.0001, 'gpus', gpus, 'expDir', expdir,...
                    'continue', true, 'numEpochs', 1500) ;

% Decrease more (but makes little difference)
[net, info] = cnn_train(net, imdb, @getBatch, 'train', -1*ones(40*batch_size,1),...
                    'val', -1*ones(1*batch_size,1), 'batchSize', batch_size,...
                    'learningRate', 0.00001, 'gpus', gpus, 'expDir', expdir,...
                    'continue', true, 'numEpochs', 1600) ;

end

% Girschick uses
%     batch_size: 128
%     crop_size: 227
%     mirror: true
%     fg_threshold: 0.5
%     bg_threshold: 0.5
%     fg_fraction: 0.25

function [imdata, labels] = getBatch(imdb, batch)
    windows = imdb.windows ;
    nclasses = size(imdb.classes, 1) ;
    
    batch_size = length(batch) ;
    fg_thresh = 0.5 ;
    bg_thresh = 0.5 ;
    fg_proportion = 0.25 ;
    crop_padding = 16 ;
    
    nfg = 0 ;
    
    imdata = zeros(227, 227, 3, batch_size, 'single') ;
    labels = zeros(1, batch_size) ;
    
    % Create batch of windows with specified
    % proportion of foreground/background windows
    batch_windows = zeros(batch_size, 7) ;
    for b=1:batch_size
        want_fg = rand <= fg_proportion ;
        while true
            rand_index = randi([1,size(windows, 1)]) ;
            ov = windows(rand_index, 2) ;
            if ov >= fg_thresh && want_fg
                batch_windows(b,:) = windows(rand_index, :) ;
                break
            elseif ov < bg_thresh && ~want_fg
                batch_windows(b,:) = [nclasses+1 windows(rand_index, 2:end)] ;
                break
            end
        end
        
        nfg = nfg + want_fg ;
    end
    
    % Extract the actual windows from the image
    for i=1:size(batch_windows, 1)
        bb = batch_windows(i, 3:6) ;
        im = imread(sprintf(imdb.imgpath, imdb.image_ids{batch_windows(i, 7)})) ;
        im = single(im) ;
        crop = rcnn_crop_and_preprocess(im, bb(1), bb(2), bb(3), bb(4),...
                          crop_padding, imdb.averageImage) ;
        if rand > 0.5, crop = fliplr(crop);, end % mirror
        imdata(:,:,:,i) = crop ;
        labels(i) = batch_windows(i, 1) ;
    end
end


function ov = overlap(bbs, bb)
    % area intersection / area union
    inter = [max(bbs(:,1),bb(1)) max(bbs(:,2),bb(2)) ...
        min(bbs(:,3),bb(3))  min(bbs(:,4),bb(4))] ;
    w = inter(:,3) - inter(:,1) + 1 ;
    h = inter(:,4) - inter(:,2) + 1 ;
    inter_area = w .* h ;
    a1 = (bbs(:,3) - bbs(:,1) + 1) .* (bbs(:,4) - bbs(:,2) + 1) ;
    a2 = (bb(3) - bb(1) + 1) * (bb(4) - bb(2) + 1) ;
    union_area = a1 + a2 - inter_area ;
    ov = inter_area ./ union_area ;
    ov(w <= 0 | h <= 0) = 0 ;
end
