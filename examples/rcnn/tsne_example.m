VOC_DIR = '/data/jdt/VOCdevkit' ;
VOC_EDITION = 'VOC2007' ;
VOC_IMAGESET = 'test' ;

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;
addpath(fullfile(fileparts(mfilename('fullpath')),'..')) ;

%% Setup Net
v3 = load('/nfs/shl5.data/jdt/exp-voc2007-trainval-v3/net-epoch-1600.mat') ;
averageImage = v3.net.normalization.averageImage ;
v3.net.layers(end) = [] ;
v3.net.layers(end) = [] ;
assert(strcmp(v3.net.layers{end}.name, 'relu7'));
net = vl_simplenn_move(v3.net, 'gpu') ;

%% Load VOC test
% Load windows
[windows, image_ids, VOCopts] = rcnn_get_voc_windows(VOC_DIR, VOC_EDITION, VOC_IMAGESET) ;
% columns: class, overlap, bb1, bb2, bb3, bb4, imageid
gt_wins = windows(windows(:,2) == 1, :) ;

imdata = zeros(227, 227, 3, size(gt_wins, 1), 'single') ;
labels = zeros(1, size(gt_wins, 1)) ;

% Extract the actual windows from the image
try
    load('./tsne_gt_imdata.mat') ;
catch
    parfor i=1:size(gt_wins, 1)
        bb = gt_wins(i, 3:6) ;
        im = imread(sprintf(VOCopts.imgpath, image_ids{gt_wins(i, 7)})) ;
        im = single(im) ;
        crop = rcnn_crop_and_preprocess(im, bb(1), bb(2), bb(3), bb(4),...
            16, averageImage) ;
        %if rand > 0.5, crop = fliplr(crop); end % mirror
        imdata(:,:,:,i) = crop ;
        %labels(i) = gt_wins(i, 1) ;
    end
end

labels = gt_wins(:, 1) ;

%% Get feats from network
feats = zeros(size(gt_wins, 1), 4096, 'single') ;

for i=1:256:size(gt_wins, 1)
    batchEnd = min(size(gt_wins, 1), i+256-1) ;
    fprintf('%d : %d\n', i, batchEnd) ;
    res = vl_simplenn(net, gpuArray(imdata(:,:,:,i:batchEnd)));
    feats(i:batchEnd, :) = squeeze(gather(res(end).x))' ;
end

%% Do t-SNE
addpath('~/Code/tsne/')

t = tsne(feats, labels);

gscatter(t(:,1), t(:,2), VOCopts.classes(labels));
title('VOC2007 test, feats from netv3 epoch1600 fc7') ;

%% Load youtube-objs
addpath('/nfs/shl5.data/jdt/youtube-objects/vo-release/code/') ;
params = initVideoObjectOptions(1, true) ;
%params.videos.maskfile = 'test.txt' ;
youtube_gts = cell(size(VOCopts.classes, 1), 1) ;
youtube_imfiles = cell(size(VOCopts.classes, 1), 1) ;
nYoutube = 0;
for c=1:size(VOCopts.classes)
    class = VOCopts.classes{c} ;
    try
        [shots, info] = ap_VOGetVideos(class, params);
        disp(class) ;
    catch
        continue
    end
    gts = [] ;
    imfiles = {} ;
    for i = 1:size(shots, 2)
        gt = info(i).gt ;
        
        for f=1:size(gt, 2)
            if ~isempty(gt{f})
                gts = [gts ; i f gt{f}' c] ;% shot frame b1 b2 b3 b4 class
                d = load ([shots{i} '/dataset.mat']);
                fn = [shots{i} '/' d.dataset(f).file] ;
                imfiles = [imfiles ; fn ] ;
                nYoutube = nYoutube + 1 ;
                
            end
        end
    end
    disp(size(gts, 1)) ;
    youtube_gts{c} = gts ;
    youtube_imfiles{c} = imfiles ;
end
%%
vidimdata = zeros(227, 227, 3, nYoutube, 'single') ;
d = 1 ;
for c=1:size(VOCopts.classes)
    gts = youtube_gts{c} ;
    imfiles = youtube_imfiles{c} ;
    if ~isempty(gts), disp(VOCopts.classes{c}); disp(size(gts, 1)), end
    for i = 1:size(gts, 1)
        bb = gts(i, 3:6) ;
        shot = gts(i, 1) ;
        frame = gts(i, 2) ;
        im = imread(imfiles{i}) ;
        im = single(im) ;
        if length(size(im)) == 2
            im = cat(3, im, im, im) ;
        end
        crop = rcnn_crop_and_preprocess(im, bb(1), bb(2), bb(3), bb(4),...
            16, averageImage) ;
        %if rand > 0.5, crop = fliplr(crop); end % mirror
        vidimdata(:,:,:,d) = crop ;
        d = d + 1 ;
    end
end
ytall = cat(1, youtube_gts{:}) ;
youtube_labels = ytall(:, 7) ;

%% Get feats from network
youtube_feats = zeros(nYoutube, 4096, 'single') ;
for i=1:256:nYoutube
    batchEnd = min(nYoutube, i+256-1) ;
    fprintf('%d : %d\n', i, batchEnd) ;
    res = vl_simplenn(net, gpuArray(vidimdata(:,:,:,i:batchEnd)));
    youtube_feats(i:batchEnd, :) = squeeze(gather(res(end).x))' ;
end

%% Prepare
classesInYoutube = find(cellfun(@(x) ~isempty(x), youtube_gts)) ;
mask = zeros(size(labels)) ;
for i=classesInYoutube'
    mask = mask | (labels == i) ;
end
vocFeatsWithYoutubeClasses = feats(mask, :) ;
vocLabelsWithYoutubeClasses = labels(mask) ;

tsneX = cat(1, youtube_feats, vocFeatsWithYoutubeClasses) ;
tsneLabels = cat(1, youtube_labels, 20+vocLabelsWithYoutubeClasses) ;
tsneLabelNames = cat(1, strcat(VOCopts.classes(youtube_labels), ' yt'), VOCopts.classes(vocLabelsWithYoutubeClasses)) ;

%% do tsne

t_yt = tsne(tsneX, []) ;

%% plot
cc = colorcube;
colours = [cc(1:10, :); cc(1:10,:)] ;
markers = [repmat('x', [10,1]); repmat('.', [10,1])] ;
sizes = [repmat(6, [10,1]); repmat(4, [10,1])] ;


h = gscatter(t_yt(:,1), t_yt(:,2), tsneLabels, colours, markers,sizes, false) ;
legend(h(1:10), VOCopts.classes(classesInYoutube)')

gscatter(t_yt(:,1), t_yt(:,2), ...
    [repmat('ytb', [length(youtube_labels),1]); repmat('voc', [length(vocLabelsWithYoutubeClasses),1])]) ;