function [ windows, files ] = rcnn_get_youtube_windows( )

addpath('/nfs/shl5.data/jdt/youtube-objects/vo-release/code/') ;

classes={'aeroplane' 'bicycle' 'bird' 'boat' 'bottle' 'bus' 'car' 'cat' 'chair'...
         'cow' 'diningtable' 'dog' 'horse' 'motorbike' 'person' 'pottedplant'...
         'sheep' 'sofa' 'train' 'tvmonitor'}';
     
params = initVideoObjectOptions(1, true) ;
params.videos.maskfile = 'train.txt' ;
nYoutube = 0;
% Use the tubes to get candidate windows
for c=1:size(classes, 1)
    class = classes{c} ;
    try
        [shots, info] = ap_VOGetVideos(class, params);
        disp(class) ;
    catch
        continue
    end
    wins = [] ;
    imfiles_by_shot = {} ;
    for si = 1:size(shots, 2)
        % gt = info(si).gt ;
        d = load ([shots{si} '/dataset.mat']);
        t = load ([shots{si} '/' params.trackname], 'T');
        tubes = t.T ;
        frameimages = d.dataset ;
        imfiles_by_shot = [imfiles_by_shot ; strcat(shots{si}, '/', cat(1,frameimages.file))] ;
        
        nframes = size(d.dataset, 2) ;
        for ti=1:size(tubes, 1)
            for f=1:nframes
                box = tubes{ti,f} ;
                if ~isempty(box)
                    wins = [wins ; c 1 box si f] ;
                    % class, fakeoverlap, bb1, bb2, bb3, bb4, shoti, framei
                    
                    %fn = [shots{si} '/' frameimages(f).file] ;
                    nYoutube = nYoutube + 1 ;
                    
                end
            end
        end
    end
    disp(size(wins, 1)) ;
    ytwindows{c} = wins ;
    files{c} = imfiles_by_shot ;
end

windows = cat(1, ytwindows{:}) ;



end

