function rcnn_finetune_hinge
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
net.layers{end-1}.weights{3} = 0.01 * randn([1 1 4096 21], 'single') ;
net.layers{end-1}.weights{4} = 0.01 * randn([1 21], 'single') ;

net.layers{end-1}.learningRate = single([10 20 10 20]) ;

net.layers = net.layers(1:end-1) ; % Remove softmax layer
%net.layers{end}.type = 'softmaxloss' ;

batch_size = 128 ;
expdir = '/data/jdt/exp-voc2007-trainval-v5-hinge' ;
expdir = '/tmp/exp-voc2007' ;
gpus = [] ;

imdb.windows = windows ;
imdb.classes = VOCopts.classes ;
imdb.imgpath = VOCopts.imgpath ;
imdb.image_ids = image_ids ;
imdb.averageImage = net.normalization.averageImage ;

% Give fake train and val sets of an appropriate size, since currently
% our getBatch function just generates a batch on the fly from the
% set of windows
[net, info] = cnn_train(net, imdb, @rcnn_get_batch, 'train', -1*ones(40*batch_size,1),...
                    'val', -1*ones(1*batch_size,1), 'batchSize', batch_size,...
                    'learningRate', 0.001, 'gpus', gpus, 'expDir', expdir,...
                    'continue', false, 'numEpochs', 900) ;

end

function resnext = forward_loss_split(l, res, resnext)
    fc8_for_softmax = vl_nnconv(res.x, l.weights{1}, l.weights{2}, 'pad', l.pad, 'stride', l.stride) ;
    fc8_for_hinge = vl_nnconv(res.x, l.weights{3}, l.weights{4}, 'pad', l.pad, 'stride', l.stride) ;
    resnext.aux = struct('fc8_for_softmax', fc8_for_softmax, 'fc8_for_hinge', fc8_for_hinge) ;
    
    sm = vl_nnsoftmaxloss(fc8_for_softmax, l.class) ;
    hl = vl_nnhingeloss(fc8_for_hinge, l.class) ;
    
    err_sm = top1err(fc8_for_softmax, l.class, size(res.x, 4)) ;
    err_h = top1err(fc8_for_hinge, l.class, size(res.x, 4)) ;
    
    fprintf('\nBatch: Loss sm %f, Loss hinge %f, Err sm %f, Err hinge %f\n', ...
            sm, hl, err_sm, err_h) ;
    
    resnext.x = hl ;
end

function res = backward_loss_split(l, res, resnext)
    sm_dzdx = vl_nnsoftmaxloss(resnext.aux.fc8_for_softmax, l.class, resnext.dzdx) ;
    h_dzdx = vl_nnhingeloss(resnext.aux.fc8_for_hinge, l.class, resnext.dzdx) ;
        
    C = 0.1 ;
    h_dzdx = C * h_dzdx ;
        
    [dzdx_a, res.dzdw{1}, res.dzdw{2}] = vl_nnconv(res.x, l.weights{1}, l.weights{2}, sm_dzdx, 'pad', l.pad, 'stride', l.stride) ;
    [dzdx_b, res.dzdw{3}, res.dzdw{4}] = vl_nnconv(res.x, l.weights{3}, l.weights{4}, h_dzdx, 'pad', l.pad, 'stride', l.stride) ;
    
    % hack for error -- cnn_train expects prediction to be in res(end-1)
    res.x = resnext.aux.fc8_for_hinge ;
    
    res.dzdx = dzdx_a + dzdx_b ;
end

function err = top1err(x, classes, batchSize)
    [~,predictions] = sort(gather(x), 3, 'descend') ;
    error = ~bsxfun(@eq, predictions, reshape(classes, 1, 1, 1, [])) ;
    err = sum(sum(sum(error(:,:,1,:))))/batchSize ;
end