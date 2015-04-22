// @file vl_selectivesearch.cu
// @brief Selective Search MEX wrapper
// @author James Thewlis

/*
Copyright (C) 2015 Andrea Vedaldi and James Thewlis.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include <vector>

#include "bits/selectivesearch.hpp"
#include "bits/mexutils.h"

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  if (nin < 1) {
    mexErrMsgTxt("No image given");
  }

  if (!mxIsClass(in[0], "single")) {
    mexErrMsgTxt("Input must be single");
  }

  if (mxGetNumberOfDimensions(in[0]) != 3 || mxGetDimensions(in[0])[2] != 3) {
    mexErrMsgTxt("Input must be HxWx3");
  }

  mwSize const *dims = mxGetDimensions(in[0]);
  mwSize height = dims[0];
  mwSize width = dims[1];
  mwSize nchannels = dims[2];

  float const *data = (float *) mxGetData(in[0]);

  std::vector<int> rects;
  vl::selectivesearch(rects, data, height, width);

  int nRects = rects.size() / 4;
  out[0] = mxCreateDoubleMatrix(nRects, 4, mxREAL);
  // Make column major and +1 for matlab indexing
  double *rectsOut = mxGetPr(out[0]);
  for (int i = 0; i < nRects; ++i) {
    rectsOut[i + nRects * 0] = (double) rects[i * 4] + 1;
    rectsOut[i + nRects * 1] = (double) rects[i * 4 + 1] + 1;
    rectsOut[i + nRects * 2] = (double) rects[i * 4 + 2] + 1;
    rectsOut[i + nRects * 3] = (double) rects[i * 4 + 3] + 1;
  }
}
