function [imdata, labels] = rcnn_get_batch(imdb, batch, preloaded)

% Girschick uses
%     batch_size: 128
%     crop_size: 227
%     mirror: true
%     fg_threshold: 0.5
%     bg_threshold: 0.5
%     fg_fraction: 0.25

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
        if exist('preloaded', 'var')
            im = preloaded{batch_windows(i, 7)};
        else
            im = imread(sprintf(imdb.imgpath, imdb.image_ids{batch_windows(i, 7)})) ;
            im = single(im) ;
        end
        crop = rcnn_crop_and_preprocess(im, bb(1), bb(2), bb(3), bb(4),...
                          crop_padding, imdb.averageImage) ;
        if rand > 0.5, crop = fliplr(crop);, end % mirror
        imdata(:,:,:,i) = crop ;
        labels(i) = batch_windows(i, 1) ;
    end
end
