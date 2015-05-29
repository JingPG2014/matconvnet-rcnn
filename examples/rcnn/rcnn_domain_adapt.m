function rcnn_domain_adapt
% Finetune the CAFFE reference model to VOC2007
% First you must run utils/import-ref-models.sh
% to download and convert the model

VOC_DIR = '~/Code/VOCdevkit' ;
VOC_EDITION = 'VOC2007' ;
VOC_IMAGESET = 'trainval' ;

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;
addpath(fullfile(fileparts(mfilename('fullpath')),'..')) ;

% Load windows
[windows, image_ids, VOCopts] = rcnn_get_voc_windows(VOC_DIR, VOC_EDITION, VOC_IMAGESET) ;
[youtube_windows, youtube_files] = rcnn_get_youtube_windows() ;

% Load net
cafferef = fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'data', 'models', 'imagenet-caffe-ref.mat') ;

net = load(cafferef) ;

% Replace fc8 with out hinge/softmax split
assert(all(net.layers{end-1}.name == 'fc8')) ;

net.layers{end-1}.name = 'loss-split' ;
net.layers{end-1}.type = 'custom' ;
net.layers{end-1}.forward = @forward_loss_split ;
net.layers{end-1}.backward = @backward_loss_split ;


net.layers{end-1}.weights{1} = 0.01 * randn([1 1 4096 21], 'single') ;
net.layers{end-1}.weights{2} = 0.01 * randn([1 21], 'single') ;
net.layers{end-1}.weights{3} = 0.01 * randn([1 1 4096 2], 'single') ;
net.layers{end-1}.weights{4} = 0.01 * randn([1 2], 'single') ;

net.layers{end-1}.learningRate = single([10 20 10 20]) ;

net.layers = net.layers(1:end-1) ; % Remove softmax layer
%net.layers{end}.type = 'softmaxloss' ;

batch_size = 256 ;
expdir = '/data/jdt/exp-voc2007-trainval-v5-hinge' ;
expdir = '/tmp/asdf' ;
gpus = [] ;

imdb.windows = windows ;
imdb.classes = VOCopts.classes ;
imdb.imgpath = VOCopts.imgpath ;
imdb.image_ids = image_ids ;
imdb.averageImage = net.normalization.averageImage ;

imdb.youtube_windows = youtube_windows ;
imdb.youtube_files = youtube_files ;

% Give fake train and val sets of an appropriate size, since currently
% our getBatch function just generates a batch on the fly from the
% set of windows
[net, info] = cnn_train(net, imdb, @domainadapt_get_batch, 'train', -1*ones(40*batch_size,1),...
                    'val', -1*ones(1*batch_size,1), 'batchSize', batch_size,...
                    'learningRate', 0.001, 'gpus', gpus, 'expDir', expdir,...
                    'continue', false, 'numEpochs', 900) ;

end

function resnext = forward_loss_split(l, res, resnext)
    x_cls = res.x(:,:,:,1:128) ;
    fc8_for_cls = vl_nnconv(x_cls, l.weights{1}, l.weights{2}, 'pad', l.pad, 'stride', l.stride) ;
    fc8_for_dom = vl_nnconv(res.x, l.weights{3}, l.weights{4}, 'pad', l.pad, 'stride', l.stride) ;
    resnext.aux = struct('fc8_for_cls', fc8_for_cls, 'fc8_for_dom', fc8_for_dom) ;
    
    labels_cls = l.class(1:128) ;
    labels_dom = [ones(1,128), 2*ones(1,128)] ;
    
    % x2 since we only use half the images in the batch
    % and cnn_train divides by batch size
    loss_cls = 2 * vl_nnsoftmaxloss(fc8_for_cls, labels_cls) ;
    loss_dom = vl_nnsoftmax(fc8_for_dom, labels_dom) ;
    
    err_cls = top1err(fc8_for_cls, labels_cls, size(res.x, 4)) ;
    err_dom = top1err(fc8_for_dom, labels_dom, size(res.x, 4)) ;
    
    fprintf('\nBatch: Loss cls %f, Loss dom %f, Err cls %f, Err dom %f\n', ...
            loss_cls, loss_dom, err_cls, err_dom) ;
    
    resnext.x = loss_cls ;
end

function res = backward_loss_split(l, res, resnext)
    dzdx_loss_cls = 2 * vl_nnsoftmaxloss(resnext.aux.fc8_for_cls, l.class, resnext.dzdx) ;
    dzdx_loss_dom = vl_nnsoftmaxloss(resnext.aux.fc8_for_dom, l.class, resnext.dzdx) ;
        
    [dzdx_cls, res.dzdw{1}, res.dzdw{2}] = vl_nnconv(res.x, l.weights{1}, l.weights{2}, dzdx_loss_cls, 'pad', l.pad, 'stride', l.stride) ;
    [dzdx_dom, res.dzdw{3}, res.dzdw{4}] = vl_nnconv(res.x, l.weights{3}, l.weights{4}, dzdx_loss_dom, 'pad', l.pad, 'stride', l.stride) ;
    
    % hack for error -- cnn_train expects prediction to be in res(end-1)
    res.x = resnext.aux.fc8_for_cls ;
    
    res.dzdx = dzdx_cls + dzdx_dom ;
end

function err = top1err(x, classes, batchSize)
    [~,predictions] = sort(gather(x), 3, 'descend') ;
    error = ~bsxfun(@eq, predictions, reshape(classes, 1, 1, 1, [])) ;
    err = sum(sum(sum(error(:,:,1,:))))/batchSize ;
end

function [imdata, labels] = domainadapt_get_batch(imdb, batch)
    [vocimdata, voclabels] = rcnn_get_batch(imdb, batch(1:128)) ;
    
    ytwindows = imdb.youtube_windows ;
    ytfiles = imdb.youtube_files ;
    
    yt_randIdxs = randi([1,size(ytwindows, 1)], 1, 128) ;
    
    vidimdata = zeros(227, 227, 3, 128, 'single') ;
    i = 1;
    for wi=yt_randIdxs
        cls = ytwindows(wi, 1) ;
        sh = ytwindows(wi, 7) ;
        fr = ytwindows(wi, 8) ;
        
        fn = ytfiles{cls}{sh}(fr,:) ;
        
        bb = ytwindows(wi, 3:6) ;
        im = imread(fn) ;
        im = single(im) ;
        crop = rcnn_crop_and_preprocess(im, bb(1), bb(2), bb(3), bb(4),...
            16, imdb.averageImage) ;
        if rand > 0.5, crop = fliplr(crop); end % mirror
        vidimdata(:,:,:,i) = crop ;
        i = i + 1 ;
    end
    
    imdata = cat(4, vocimdata, vidimdata) ;
    labels = [voclabels, -1*ones(1,128)] ;
end