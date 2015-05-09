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
#include <string>

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
  if (nin > 1) {
    threshConst = (float) mxGetPr(in[1])[0];
  }

  int minSize = 200;
  if (nin > 2) {
    minSize = (int) mxGetPr(in[2])[0];
  }

  // The set of similarity measures is specified as a cell
  // array of strings. A different grouping of the initial regions
  // is done for each similarity measure. A similarity measure
  // can be composed of multiple similarity functions,
  // with a character mnemonic for each function.
  // Similarity functions in the same string will be averaged.
  // Eg. {'cts','ct'} will have one similarity measure accounting
  // for colour, texture and size and another for just colour and
  // texture.
  // For C++ this is converted to an std::vector of bitwise flags.
  std::vector<int> similarityMeasures;
  if (nin > 3) {
    if (!mxIsClass(in[3], "cell")) {
      mexErrMsgTxt("Similarities must be cell array of strings");
    }
    const size_t nElems = mxGetNumberOfElements(in[3]);
    for (int i = 0; i < nElems; ++i) {
      const mxArray *cell = mxGetCell(in[3], i);
      if (!mxIsClass(cell, "char")) {
        mexErrMsgTxt("Similarities must be cell array of strings");
      }
      const size_t nChars = mxGetNumberOfElements(cell);
      const mxChar *string = (mxChar *) mxGetData(cell);
      int sim = 0;
      for (int j = 0; j < nChars; ++j) {
        char c = (char) string[j];
        switch (c) {
        case 'c': sim |= vl::SIM_COLOUR; break;
        case 't': sim |= vl::SIM_TEXTURE; break;
        case 's': sim |= vl::SIM_SIZE; break;
        case 'f': sim |= vl::SIM_FILL; break;
        default: mexWarnMsgTxt((std::string("Unknown similarity ") + c).c_str());
        }
      }
      if (sim != 0) similarityMeasures.push_back(sim);
    }
  } else {
    similarityMeasures.push_back(vl::SIM_COLOUR | vl::SIM_TEXTURE | vl::SIM_SIZE | vl::SIM_FILL);
    similarityMeasures.push_back(vl::SIM_TEXTURE | vl::SIM_SIZE | vl::SIM_FILL);
  }

  if (similarityMeasures.size() == 0) {
    mexErrMsgTxt("No valid similarity measures given");
  }


  mwSize const *dims = mxGetDimensions(in[0]);
  mwSize height = dims[0];
  mwSize width = dims[1];
  mwSize nchannels = dims[2];

  float const *data = (float *) mxGetData(in[0]);

  std::vector<int> rects;

  std::vector<int> initSeg;
  std::vector<float> histTexture;
  std::vector<float> histColour;

  // Allow providing initial segmentation and histograms, for debugging

  if (nin > 4) {
    assert(mxIsClass(in[4], "int32"));
    assert(width*height == mxGetDimensions(in[4])[0] * mxGetDimensions(in[4])[1]);
    initSeg = std::vector<int>((int *) mxGetData(in[4]), ((int *) mxGetData(in[4])) + width*height);
  }

  if (nin > 5) {
    assert(mxIsClass(in[5], "single"));
    int n = mxGetDimensions(in[5])[0] * mxGetDimensions(in[5])[1];
    histTexture = std::vector<float>((float *) mxGetData(in[5]), ((float *) mxGetData(in[5])) + n);
  }

  if (nin > 6) {
    assert(mxIsClass(in[6], "single"));
    int n = mxGetDimensions(in[6])[0] * mxGetDimensions(in[6])[1];
    histColour = std::vector<float>((float *) mxGetData(in[6]), ((float *) mxGetData(in[6])) + n);
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
