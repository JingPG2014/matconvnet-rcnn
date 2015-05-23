function rcnn_finetune
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

% Adapt fc8 to pascal
assert(all(net.layers{end-1}.name == 'fc8')) ;

net.layers{end-1}.weights{1} = 0.01 * randn([1 1 4096 21], 'single') ;
net.layers{end-1}.weights{2} = 0.01 * randn([1 21], 'single') ;
net.layers{end-1}.learningRate = single([10 20]) ;

net.layers{end}.type = 'softmaxloss' ;

batch_size = 128 ;
%expdir = '/data/jdt/exp-voc2007-trainval-v3' ;
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
                    'continue', true, 'numEpochs', 900) ;

% Decrease learning rate
[net, info] = cnn_train(net, imdb, @rcnn_get_batch, 'train', -1*ones(40*batch_size,1),...
                    'val', -1*ones(1*batch_size,1), 'batchSize', batch_size,...
                    'learningRate', 0.0001, 'gpus', gpus, 'expDir', expdir,...
                    'continue', true, 'numEpochs', 1500) ;

% Decrease more (but makes little difference)
[net, info] = cnn_train(net, imdb, @rcnn_get_batch, 'train', -1*ones(40*batch_size,1),...
                    'val', -1*ones(1*batch_size,1), 'batchSize', batch_size,...
                    'learningRate', 0.00001, 'gpus', gpus, 'expDir', expdir,...
                    'continue', true, 'numEpochs', 1600) ;

end





