function out = rcnn_crop_and_preprocess(im, x1, y1, x2, y2, padding, meanim)
    % Crop as in https://github.com/BVLC/caffe/blob/master/src/caffe/layers/window_data_layer.cpp#L302
    out_size = size(meanim, 1) ;
    
    % scale box such that we have 'padding' pixels on each side
    % after resizing
    context_scale = out_size / (out_size - 2 * padding) ;
    
    % selective search box dims
    h = (y2 - y1 + 1) ;
    w = (x2 - x1 + 1) ;
    
    % centre
    cx = x1 + w/2 ;
    cy = y1 + h/2 ;
    
    % coords of box when expanded with padding
    x1p = round(cx - w/2 * context_scale) ;
    y1p = round(cy - h/2 * context_scale) ;
    x2p = round(cx + w/2 * context_scale) ;
    y2p = round(cy + h/2 * context_scale) ;
    pboxw = length(x1p:x2p) ;
    pboxh = length(y1p:y2p) ;

    % coords of box with padding clipped to image dims
    x1c = max(1, x1p) ;
    y1c = max(1, y1p) ;
    x2c = min(size(im,2), x2p) ;
    y2c = min(size(im,1), y2p) ;
    
    boxim = im(y1c:y2c, x1c:x2c, :) ;
    cropw = round(size(boxim, 2) * out_size/pboxw) ;
    croph = round(size(boxim, 1) * out_size/pboxh) ;
    
    % Offset at which to place clipped box in output image
    xoffs = round((x1c - x1p) * out_size/length(x1p:x2p)) ;
    yoffs = round((y1c - y1p) * out_size/length(y1p:y2p)) ;
    
    % Enforce that the clipped box does not exceed the output dimensions
    % due to rounding
    if cropw + xoffs > out_size
        cropw = out_size - xoffs ;
    end
    if croph + yoffs > out_size
        croph = out_size - yoffs ;
    end
    
    % Girshick turns of antialiasing to be like OpenCV
    boxim = imresize(boxim, [croph cropw], 'bilinear', 'antialiasing', false) ;
    
    % Subtract mean
    boxim = boxim - meanim(yoffs+1:yoffs+croph, xoffs+1:xoffs+cropw, :) ;
    
    out = zeros(size(meanim), 'single') ;

    out(yoffs+1:yoffs+croph, xoffs+1:xoffs+cropw, :) = boxim ;
end