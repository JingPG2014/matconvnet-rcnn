function rcnn_domain_adapt(initlam)
% Finetune the CAFFE reference model to VOC2007
% First you must run utils/import-ref-models.sh
% to download and convert the model

if(~isempty(initlam))
   backward_loss_split(initlam) ; 
end

VOC_DIR = '/data/jdt/VOCdevkit' ;
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
expdir = '/data/jdt/exp-domainadapt-v2' ;
%expdir = '/tmp/asdff' ;
gpus = [1] ;

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
[net, info] = cnn_train(net, imdb, @domainadapt_get_batch, 'train', -1*ones(400*batch_size,1),...
    'val', -1*ones(1*batch_size,1), 'batchSize', batch_size,...
    'learningRate', 0.001, 'gpus', gpus, 'expDir', expdir,...
    'continue', true, 'numEpochs', 90) ;
[net, info] = cnn_train(net, imdb, @domainadapt_get_batch, 'train', -1*ones(400*batch_size,1),...
    'val', -1*ones(1*batch_size,1), 'batchSize', batch_size,...
    'learningRate', 0.0001, 'gpus', gpus, 'expDir', expdir,...
    'continue', true, 'numEpochs', 150) ;
[net, info] = cnn_train(net, imdb, @domainadapt_get_batch, 'train', -1*ones(400*batch_size,1),...
    'val', -1*ones(1*batch_size,1), 'batchSize', batch_size,...
    'learningRate', 0.00001, 'gpus', gpus, 'expDir', expdir,...
    'continue', true, 'numEpochs', 160) ;
                

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
    loss_dom = vl_nnsoftmaxloss(fc8_for_dom, labels_dom) ;
    
    if isnan(loss_dom)
        error 'NaN loss'
    end
    
    err_cls = top1err(fc8_for_cls, labels_cls, size(res.x, 4)) ;
    err_dom = top1err(fc8_for_dom, labels_dom, size(res.x, 4)) ;
    
    fprintf('\nBatch: Loss cls %f, Loss dom %f, Err cls %f, Err dom %f\n', ...
            loss_cls, loss_dom, err_cls, err_dom) ;
    
    resnext.x = loss_cls ;
    resnext.aux.loss_dom = loss_dom ;
end

function res = backward_loss_split(l, res, resnext)
    persistent lambda ; % ugh
    if(nargin == 1)
        lambda = l;
        return
    end
    labels_cls = l.class(1:128) ;
    labels_dom = [ones(1,128), 2*ones(1,128)] ;
    
    dzdx_loss_cls = 2 * vl_nnsoftmaxloss(resnext.aux.fc8_for_cls, labels_cls, resnext.dzdx) ;
    dzdx_loss_dom = vl_nnsoftmaxloss(resnext.aux.fc8_for_dom, labels_dom, resnext.dzdx) ;
        
    [dzdx_cls, res.dzdw{1}, res.dzdw{2}] = vl_nnconv(res.x(:,:,:,1:128), l.weights{1}, l.weights{2}, dzdx_loss_cls, 'pad', l.pad, 'stride', l.stride) ;
    [dzdx_dom, res.dzdw{3}, res.dzdw{4}] = vl_nnconv(res.x, l.weights{3}, l.weights{4}, dzdx_loss_dom, 'pad', l.pad, 'stride', l.stride) ;
    
    % hack for error -- cnn_train expects prediction to be in res(end-1)
    res.x = cat(4, resnext.aux.fc8_for_cls,  resnext.aux.fc8_for_cls) ;
    
    if isempty(lambda), lambda = 0; end
        
    lambda = min(lambda + 0.0001, 1);
    lam = lambda;
    %if rand > .50, lam = 0; end
    
    if (resnext.aux.loss_dom > 400), lam = 0; end
    
    fprintf(' lambda : %f ', lam) ;

    res.dzdx = cat(4, dzdx_cls, zeros(1,1,4096,128)) + -lam * dzdx_dom ;
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
    for i=1:128
        while 1
        wi = randi([1, size(ytwindows,1)]);
        cls = ytwindows(wi, 1) ;
        sh = ytwindows(wi, 7) ;
        fr = ytwindows(wi, 8) ;
        
        fn = ytfiles{cls}{sh}(fr,:) ;
        
        bb = ytwindows(wi, 3:6) ;
        if any(isnan(bb)), disp('NaN bb???'); continue; end
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
        crop = rcnn_crop_and_preprocess(im, bb(1), bb(2), bb(3), bb(4),...
            16, imdb.averageImage) ;
        if rand > 0.5, crop = fliplr(crop); end % mirror
        vidimdata(:,:,:,i) = crop ;
        i = i + 1 ;
    end
    
    imdata = cat(4, vocimdata, vidimdata) ;
    labels = [voclabels, voclabels] ; % hack for cnn_train
end
