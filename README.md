# R-CNN Implementation for MatConvnet

An implementation of selective search[1] and R-CNN[2] for the [MatConvnet](http://www.vlfeat.org/matconvnet/) CNN library.

Code in [examples/rcnn](https://github.com/jamt9000/matconvnet-rcnn/tree/selectivesearch/examples/rcnn)

For the Selective Search implementation see [vl_regionproposals](https://github.com/jamt9000/matconvnet-rcnn/blob/selectivesearch/matlab/vl_regionproposals.m) which is a wrapper around mexfile [vl_selectivesearch.cpp](https://github.com/jamt9000/matconvnet-rcnn/blob/selectivesearch/matlab/src/vl_selectivesearch.cpp) itself wrapping the pure C++ version [selectivesearch.cpp](https://github.com/jamt9000/matconvnet-rcnn/blob/selectivesearch/matlab/src/bits/selectivesearch.cpp).


[1] Uijlings, Jasper RR, et al. "Selective search for object recognition." International journal of computer vision 104.2 (2013): 154-171.

[2] Girshick, Ross, et al. "Rich feature hierarchies for accurate object detection and semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.
