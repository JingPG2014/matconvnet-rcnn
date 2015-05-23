function [windows, image_ids, VOCopts] = rcnn_get_voc_windows(voc_dir, voc_edition, voc_imageset)
% Get ground truth and selective search windows for PASCAL VOC

% Load VOC dataset
matfile = sprintf('windows_%s_%s.mat', voc_edition, voc_imageset) ;

addpath(fullfile(voc_dir, 'VOCcode')) ;
VOCopts = [] ;
VOCinit ; % creates VOCopts
image_ids = textread(sprintf(VOCopts.imgsetpath, voc_imageset), '%s') ;
    
try
    load(matfile) ;
catch
    windows = cell(length(image_ids), 1) ;
    parfor i = 1:length(image_ids)
        disp(i)
        % Load annotation data
        rec = PASreadrecord(sprintf(VOCopts.annopath, image_ids{i})) ;
        
        % Do selective search
        proposals = vl_regionproposals(imread(fullfile(voc_dir, rec.imgname))) ;
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

imids = cell(size(windows, 1) ,1) ;
for i=1:size(windows, 1) 
    imids{i} = i * ones(size(windows{i}, 1), 1) ;
end
windows = [cat(1, windows{:}), cat(1,imids{:})] ;
% columns: class, overlap, bb1, bb2, bb3, bb4, imageid

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