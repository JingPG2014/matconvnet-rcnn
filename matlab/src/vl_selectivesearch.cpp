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

  float threshConst = 200.0f;
  if (nin >= 2) {
    threshConst = (float) mxGetPr(in[1])[0];
  }

  int minSize = 200;
  if (nin >= 3) {
    minSize = (int) mxGetPr(in[2])[0];
  }

  mwSize const *dims = mxGetDimensions(in[0]);
  mwSize height = dims[0];
  mwSize width = dims[1];
  mwSize nchannels = dims[2];

  float const *data = (float *) mxGetData(in[0]);

  std::vector<int> rects;
  std::vector<int> similarityMeasures;

  // TODO set from matlab
  similarityMeasures.push_back(vl::SIM_COLOUR | vl::SIM_TEXTURE | vl::SIM_SIZE | vl::SIM_FILL);
  similarityMeasures.push_back(vl::SIM_TEXTURE | vl::SIM_SIZE | vl::SIM_FILL);

  std::vector<int> initSeg;
  std::vector<float> histTexture;
  std::vector<float> histColour;

  // Allow providing initial segmentation and histograms, for debugging

  if (nin >= 4) {
    assert(mxIsClass(in[3], "int32"));
    assert(width*height == mxGetDimensions(in[3])[0] * mxGetDimensions(in[3])[1]);
    initSeg = std::vector<int>((int *) mxGetData(in[3]), ((int *) mxGetData(in[3])) + width*height);
  }

  if (nin >= 5) {
    assert(mxIsClass(in[4], "single"));
    int n = mxGetDimensions(in[4])[0] * mxGetDimensions(in[4])[1];
    histTexture = std::vector<float>((float *) mxGetData(in[4]), ((float *) mxGetData(in[4])) + n);
  }

  if (nin >= 6) {
    assert(mxIsClass(in[5], "single"));
    int n = mxGetDimensions(in[5])[0] * mxGetDimensions(in[5])[1];
    histColour = std::vector<float>((float *) mxGetData(in[5]), ((float *) mxGetData(in[5])) + n);
  }

  vl::selectivesearch(rects, initSeg, histTexture, histColour, data,
                      height, width, similarityMeasures, threshConst, minSize);

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

  // Output initial segmentation if requested
  if (nout >= 2) {
    out[1] = mxCreateDoubleMatrix(height, width, mxREAL);
    for (int i = 0; i < height * width; ++i) mxGetPr(out[1])[i] = (double) initSeg[i];
  }

  // Output initial histograms if requested
  if (nout >= 3) {
    out[2] = mxCreateCellMatrix(2, 1);

    mxArray *tex = mxCreateDoubleMatrix(240, histTexture.size()/240, mxREAL);
    for (int i = 0; i < histTexture.size(); ++i) mxGetPr(tex)[i] = (double) histTexture[i];
    mxArray *col = mxCreateDoubleMatrix(75, histColour.size()/75, mxREAL);
    for (int i = 0; i < histColour.size(); ++i) mxGetPr(col)[i] = (double) histColour[i];

    mxSetCell(out[2], 0, tex);
    mxSetCell(out[2], 1, col);
  }
}
