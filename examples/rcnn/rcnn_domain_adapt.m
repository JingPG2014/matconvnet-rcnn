function rcnn_domain_adapt(inititer)
% Finetune the CAFFE reference model to VOC2007
% First you must run utils/import-ref-models.sh
% to download and convert the model

if(~isempty(inititer))
   backward_loss_split(inititer) ; 
end

VOC_DIR = '~/ostdata/VOCdevkit' ;
VOC_EDITION = 'VOC2007' ;
VOC_IMAGESET = 'trainval' ;
YT_DIR = '~/ostdata/youtube-objects' ;

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;
addpath(fullfile(fileparts(mfilename('fullpath')),'..')) ;

% Load windows
[windows, image_ids, VOCopts] = rcnn_get_voc_windows(VOC_DIR, VOC_EDITION, VOC_IMAGESET) ;
[youtube_windows, youtube_files] = rcnn_get_youtube_windows(YT_DIR) ;

voc_files = cellfun(@(x) sprintf(VOCopts.imgpath, x), image_ids, 'UniformOutput', false);
vocims = vl_imreadjpeg(voc_files);

% Load net
cafferef = fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'data', 'models', 'imagenet-caffe-ref.mat') ;

%net = load('/data/jdt/exp-voc2007-trainval-v3/net-epoch-1600.mat') ;
net = load(cafferef) ;
%net = net.net ;

% Replace fc8 with bottleneck
assert(strcmp(net.layers{end-1}.name, 'fc8')) ;
net.layers{end-1}.name = 'bottleneck' ;
net.layers{end-1}.weights{1} = 0.005 * randn([1 1 4096 256], 'single') ;
net.layers{end-1}.weights{2} = 0.1 * ones([1 256], 'single') ;
net.layers{end-1}.learningRate = single([10 20]) ;
net.layers{end-1}.weightDecay = [1 0] ;

% Replace softmaxloss with split
assert(strcmp(net.layers{end}.type(1:7), 'softmax'));
net.layers{end}.name = 'loss-split' ;
net.layers{end}.type = 'custom' ;
net.layers{end}.forward = @forward_loss_split ;
net.layers{end}.backward = @backward_loss_split ;


net.layers{end}.weights{1} = 0.01 * randn([1 1 256 21], 'single') ; %fc8 clas
net.layers{end}.weights{2} = zeros([1 21], 'single') ;
net.layers{end}.weights{3} = 0.01 * randn([1 1 256 1024], 'single') ; %fc8 dom
net.layers{end}.weights{4} = zeros([1 1024], 'single') ;
net.layers{end}.weights{5} = 0.01 * randn([1 1 1024 1024], 'single') ; % fc9 dom
net.layers{end}.weights{6} = zeros([1 1024], 'single') ;
net.layers{end}.weights{7} = 0.3 * randn([1 1 1024 2], 'single') ; % fc10 dom
net.layers{end}.weights{8} = zeros([1 2], 'single') ;

net.layers{end}.learningRate = single([10 20 10 20 10 20 10 20]) ;
net.layers{end}.weightDecay = [1 1 1 1 1 1 1 1] ;
net.layers{end}.pad = [0 0 0 0];
net.layers{end}.stride = [1 1];

%net.layers = net.layers(1:end-1) ; % Remove softmax layer
%net.layers{end}.type = 'softmaxloss' ;

batch_size = 256 ;
expdir = '/data/jdt/exp-domainadapt-v2' ;
expdir = '~/ostdata/exp-domainadapt-v3f' ;
gpus = [3] ;

imdb.windows = windows ;
imdb.classes = VOCopts.classes ;
imdb.imgpath = VOCopts.imgpath ;
imdb.image_ids = image_ids ;
imdb.averageImage = net.normalization.averageImage ;

imdb.youtube_windows = youtube_windows ;
imdb.youtube_files = youtube_files ;

if exist('ytimdata.mat', 'file')
load('ytimdata', 'ytimdata');
imdb.youtube_preprocessed_windows = ytimdata;
end
imdb.voc_preloaded_ims = vocims;

lrs = 2 * 0.001./((1+10*(0:160)./160).^0.75);

% Give fake train and val sets of an appropriate size, since currently
% our getBatch function just generates a batch on the fly from the
% set of windows
[net, info] = cnn_train(net, imdb, @domainadapt_get_batch, 'train', -1*ones(400*batch_size,1),...
    'val', -1*ones(1*batch_size,1), 'batchSize', batch_size,...
    'learningRate', lrs, 'gpus', gpus, 'expDir', expdir,...
    'continue', true, 'numEpochs', 160) ;
% [net, info] = cnn_train(net, imdb, @domainadapt_get_batch, 'train', -1*ones(400*batch_size,1),...
%     'val', -1*ones(1*batch_size,1), 'batchSize', batch_size,...
%     'learningRate', 0.0001, 'gpus', gpus, 'expDir', expdir,...
%     'continue', true, 'numEpochs', 150) ;
% [net, info] = cnn_train(net, imdb, @domainadapt_get_batch, 'train', -1*ones(400*batch_size,1),...
%     'val', -1*ones(1*batch_size,1), 'batchSize', batch_size,...
%     'learningRate', 0.00001, 'gpus', gpus, 'expDir', expdir,...
%     'continue', true, 'numEpochs', 160) ;
                

end

function resnext = forward_loss_split(l, res, resnext)
    resnext.aux = struct ;
    
    % Object classifier
    x_cls = res.x(:,:,:,1:128) ;
    labels_cls = l.class(1:128) ;
    % 4096 -> 21
    fc8_for_cls = vl_nnconv(x_cls, l.weights{1}, l.weights{2}, 'pad', l.pad, 'stride', l.stride) ;
    % x2 since we only use half the images in the batch
    % and cnn_train divides by batch size
    loss_cls = 2 * vl_nnsoftmaxloss(fc8_for_cls, labels_cls) ;
    
    resnext.aux.fc8_for_cls = fc8_for_cls;
    resnext.aux.loss_cls = loss_cls ;
    
    % Domain classifier
    % fc8: 4096 -> 1024
    fc8_for_dom = vl_nnconv(res.x, l.weights{3}, l.weights{4}, 'pad', l.pad, 'stride', l.stride) ;
    fc8_for_dom = vl_nnrelu(fc8_for_dom) ;
    [fc8_for_dom_drop, fc8_for_dom_dropmask] = vl_nndropout(fc8_for_dom, 'rate', 0.5) ;
    % fc9: 1024 -> 1024
    fc9_for_dom = vl_nnconv(fc8_for_dom_drop, l.weights{5}, l.weights{6}, 'pad', l.pad, 'stride', l.stride) ;
    fc9_for_dom = vl_nnrelu(fc9_for_dom) ;
    [fc9_for_dom_drop, fc9_for_dom_dropmask] = vl_nndropout(fc9_for_dom, 'rate', 0.5) ;
    % fc10: 1024 -> 2
    fc10_for_dom = vl_nnconv(fc9_for_dom_drop, l.weights{7}, l.weights{8}, 'pad', l.pad, 'stride', l.stride) ;
    % Loss
    labels_dom = [ones(1,128), 2*ones(1,128)] ;
    % weight by 0.1
    loss_dom = .1 * vl_nnsoftmaxloss(fc10_for_dom, labels_dom) ;
    
    resnext.aux.fc8_for_dom = fc8_for_dom;
    resnext.aux.fc8_for_dom_drop = fc8_for_dom_drop;
    resnext.aux.fc8_for_dom_dropmask = fc8_for_dom_dropmask;
    resnext.aux.fc9_for_dom = fc9_for_dom;
    resnext.aux.fc9_for_dom_drop = fc9_for_dom_drop;
    resnext.aux.fc9_for_dom_dropmask = fc9_for_dom_dropmask;
    resnext.aux.fc10_for_dom = fc10_for_dom ;
    resnext.aux.loss_dom = loss_dom ;

    if isnan(loss_dom)
        error 'NaN loss'
    end
    
    err_cls = top1err(fc8_for_cls, labels_cls, size(x_cls, 4)) ;
    err_dom = top1err(fc10_for_dom, labels_dom, size(res.x, 4)) ;
    
    fprintf('\nBatch: Loss cls %f, Loss dom %f, Err cls %f, Err dom %f\n', ...
            loss_cls, loss_dom, err_cls, err_dom) ;
    
    resnext.x = loss_cls ;
end

function res = backward_loss_split(l, res, resnext)
    persistent iter ; % ugh
    if(nargin == 1)
        iter = l;
        return
    end
    aux = resnext.aux ;
    
    % Object classifier
    x_cls = res.x(:,:,:,1:128) ;
    labels_cls = l.class(1:128) ;
    dzdx_loss_cls = 2 * vl_nnsoftmaxloss(aux.fc8_for_cls, labels_cls, resnext.dzdx) ;
    [dzdx_cls, res.dzdw{1}, res.dzdw{2}] = vl_nnconv(x_cls, l.weights{1}, l.weights{2}, dzdx_loss_cls, 'pad', l.pad, 'stride', l.stride) ;
    
    % Domain classifier
    % Loss
    labels_dom = [ones(1,128), 2*ones(1,128)] ;
    dzdx_loss_dom = .1 * vl_nnsoftmaxloss(aux.fc10_for_dom, labels_dom, resnext.dzdx) ;
    % fc10: 1024 <- 2
    [dzdx_fc10_for_dom, res.dzdw{7}, res.dzdw{8}] = vl_nnconv(aux.fc9_for_dom_drop, l.weights{7}, l.weights{8}, dzdx_loss_dom, 'pad', l.pad, 'stride', l.stride) ;
    % fc9: 1024 <- 1024
    dzdx_fc9_for_dom_drop = vl_nndropout(aux.fc9_for_dom, dzdx_fc10_for_dom, 'mask', aux.fc9_for_dom_dropmask) ;
    dzdx_fc9_for_dom_relu = vl_nnrelu(aux.fc9_for_dom, dzdx_fc9_for_dom_drop) ; % relu can reuse output as input
    [dzdx_fc9_for_dom, res.dzdw{5}, res.dzdw{6}] = vl_nnconv(aux.fc8_for_dom_drop, l.weights{5}, l.weights{6}, dzdx_fc9_for_dom_relu, 'pad', l.pad, 'stride', l.stride) ;
    % fc8 4096 <- 1024
    dzdx_fc8_for_dom_drop = vl_nndropout(aux.fc8_for_dom, dzdx_fc9_for_dom, 'mask', aux.fc8_for_dom_dropmask) ;
    dzdx_fc8_for_dom_relu = vl_nnrelu(aux.fc8_for_dom, dzdx_fc8_for_dom_drop) ; % relu can reuse output as input
    [dzdx_fc8_for_dom, res.dzdw{3}, res.dzdw{4}] = vl_nnconv(res.x, l.weights{3}, l.weights{4}, dzdx_fc8_for_dom_relu, 'pad', l.pad, 'stride', l.stride) ;
    dzdx_dom = dzdx_fc8_for_dom ;

    % hack for error -- cnn_train expects prediction to be in res(end-1)
    res.x = cat(4, aux.fc8_for_cls,  aux.fc8_for_cls) ;
    

    p = min(1, iter / 15000) ; % reach 1 at roughly 1/5 of training
    lambda = 2/(1 + exp(-10 * p)) - 1 ;
    fprintf('lambda %f iter %d ', lambda, iter) ;
    iter = iter + 1;
    fprintf(' normcls = %f  normdom = %f\n', norm(gather(dzdx_cls(:))), norm(gather(dzdx_dom(:))));
    grad_scale_dom = -lambda ;
    res.dzdx = cat(4, dzdx_cls, zeros(1,1,256,128)) + grad_scale_dom * dzdx_dom;
    %res.dzdx = zeros(1,1,'single');
end

function err = top1err(x, classes, batchSize)
    [~,predictions] = sort(gather(x), 3, 'descend') ;
    error = ~bsxfun(@eq, predictions, reshape(classes, 1, 1, 1, [])) ;
    err = sum(sum(sum(error(:,:,1,:))))/batchSize ;
end

function [imdata, labels] = domainadapt_get_batch(imdb, batch)
    tvoc = tic;
    [vocimdata, voclabels] = rcnn_get_batch(imdb, batch(1:128), imdb.voc_preloaded_ims) ;
    vocbatch = toc(tvoc)
    
    ytdata_in_ram = true;
    try
        ytpreproc = imdb.youtube_preprocessed_windows;
    catch
        ytdata_in_ram = false;
    end
    
    tyt = tic;
    ytwindows = imdb.youtube_windows ;
    ytfiles = imdb.youtube_files ;
    
    yt_randIdxs = randi([1,size(ytwindows, 1)], 1, 128) ;
    
    vidimdata = zeros(227, 227, 3, 128, 'single') ;
    i = 1;
    for i=1:128
        while 1
          wi = randi([1, size(ytwindows,1)]);
          cls = ytwindows(wi, 1) ;
          sh = ytwindows(wi, 7) ;
          fr = ytwindows(wi, 8) ;
          
          
          bb = ytwindows(wi, 3:6) ;
          if any(isnan(bb)), disp('NaN bb???'); continue; end
          if ytdata_in_ram
            im = ytpreproc(:,:,:,wi);
            if all(im==0)
              disp('reject'); continue;
            else
              break
            end
          else
            fn = ytfiles{cls}{sh}(fr,:) ;
            im = imread(fn) ;
            im = single(im) ;
            if length(size(im)) == 2
                im = cat(3, im, im, im) ;
            end
            x1 = bb(1); y1=bb(2); x2=bb(3); y2=bb(4);
            if min(bb) < 1 || max([x1 x2]) > size(im,2) || max([y1 y2]) > size(im,1)
                disp('reject'); bb
            else
                break
            end
          end
        end
        if ytdata_in_ram
            crop = single(im) - single(imdb.averageImage);
        else
            crop = rcnn_crop_and_preprocess(im, bb(1), bb(2), bb(3), bb(4),...
            16, imdb.averageImage) ;
        end
        if rand > 0.5, crop = fliplr(crop); end % mirror
        vidimdata(:,:,:,i) = crop ;
        i = i + 1 ;
    end
    
    imdata = cat(4, vocimdata, vidimdata) ;
    labels = [voclabels, voclabels] ; % hack for cnn_train
    ytbatch = toc(tyt)
end
