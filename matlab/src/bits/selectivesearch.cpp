#include "selectivesearch.hpp"

#include <assert.h>
#include <algorithm>
#include <math.h>
#include <limits>
#include <vector>

static int const nchannels = 3;


/* Selective search from
* J.R.R. Uijlings, K.E.A. van de Sande, T. Gevers, and A.W.M. Smeulders.
* Selective Search for Object Recognition
* IJCV, 2013.
* */

/* Initial segmentation based on
 * Felzenszwalb, P. F., & Huttenlocher, D. P. (2004).
 * Efficient graph-based image segmentation. IJCV
 * */

// See https://en.wikipedia.org/wiki/Disjoint-set_data_structure#Disjoint-set_forests
static int findHead(int vert, std::vector<int>& connectionMap)
{
  assert(vert < connectionMap.size());
  int h = vert;
  // Traverse graph until we find vertex connected to itself
  while (connectionMap[h] != h) {
    h = connectionMap[h];
  }
  return h;
}


int joinRegion(int vertexAHead, int vertexBHead, std::vector<int>& connectionMap,
               std::vector<int>& sizeMap, std::vector<int>& rankMap)
{
  int newHead = vertexBHead;
  int oldHead = vertexAHead;
  if (rankMap[vertexAHead] >= rankMap[vertexBHead]) {
    newHead = vertexAHead;
    oldHead = vertexBHead;
  }
  connectionMap[oldHead] = newHead;
  sizeMap[newHead] += sizeMap[oldHead];
  rankMap[newHead] += (int) (rankMap[vertexAHead] == rankMap[vertexBHead]);

  return newHead;
}

static void compressPath(int vert, std::vector<int>& connectionMap)
{
  int h = vert;
  std::vector<int> path;
  while (connectionMap[h] != h) {
    path.push_back(h);
    h = connectionMap[h];
  }
  for (int i = 0; i < path.size(); ++i) {
    connectionMap[path[i]] = h;
  }
}

static bool isBoundary(float edgeWeight, int a, int b, std::vector<float>& thresholds)
{
  return edgeWeight > (std::min)(thresholds[a], thresholds[b]);
}

class SortIndices
{
  std::vector<float>& v;
public:
  SortIndices(std::vector<float>& v) : v(v) {}
  bool operator() (const int& a, const int& b) const { return v[a] < v[b]; }
};

template <typename Out, typename In>
void filter1DSymmetric(Out& output, In& data, std::vector<float>& filter,
                       int height, int width, int channel, bool filterRows)
{
  for (int j = 0; j < width; ++j) {
    for (int i = 0; i < height; ++i) {
      float s = filter[0] * data[height*width*channel + height*j + i];
      for (int f = 1; f < filter.size(); ++f) {
        int iprev = filterRows ? (std::max)(0, i - f) : i;
        int inext = filterRows ? (std::min)((int)height - 1, i + f) : i;
        int jprev = filterRows ? j : (std::max)(0, j - f);
        int jnext = filterRows ? j : (std::min)((int)width - 1, j + f);
        s += data[height*width*channel + height*jprev + iprev] * filter[f];
        s += data[height*width*channel + height*jnext + inext] * filter[f];
      }
      output[height*width*channel + height*j + i] = s;
    }
  }
}

template <typename Out, typename In>
void filter1D(Out& output, In& data, std::vector<float>& filter,
              int height, int width, int channel, bool filterRows)
{
  for (int j = 0; j < width; ++j) {
    for (int i = 0; i < height; ++i) {
      float s = 0.0f;
      for (int f = 0; f < filter.size(); ++f) {
        int offs = f - filter.size()/2;
        int ioff = filterRows ? (std::min)((int)height - 1, (std::max)(0, i + offs)) : i;
        int joff = filterRows ? j : (std::min)((int)width - 1, (std::max)(0, j + offs));
        s += data[height*width*channel + height*joff + ioff] * filter[f];
      }
      output[height*width*channel + height*j + i] = s;
    }
  }
}

void angleGrad(std::vector<float>& out, float const *smoothed, int height, int width, int channel, float phi)
{
  // Angled derivative
  // First smooth with gaussian then take derivative with [-1,0,1]
  float sinp = 0.5f * sinf(phi);
  float cosp = 0.5f * cosf(phi);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      float top = smoothed[height*width*channel + (std::max)(i-1, 0) + j * height];
      float bot = smoothed[height*width*channel + (std::min)(i+1, height-1) + j * height];
      float left = smoothed[height*width*channel + i + (std::max)(j-1, 0) * height];
      float right = smoothed[height*width*channel + i + (std::min)(j+1,width-1) * height];
      out[i + j * height] = sinp*(top-bot)+cosp*(left-right);
    }
  }
}

static void gaussianBlur(float *output, float const *data, int height, int width)
{

  // we only compute centre and positive side of gaussian, since it is symmetric
  // so effectively a 9x9 filter
  int const filterVectorSize = 5;
  float const sigma = 0.8f;

  // 1D gaussian filter (non-negative half)
  std::vector<float> filter(filterVectorSize, 0);
  float total = 0.0f;
  for (int f = 0; f < filterVectorSize; ++f) {
    filter[f] = expf(-0.5f*(f/sigma)*(f/sigma));
    total += filter[f];
  }
  total = total * 2 - filter[0];
  // Normalise
  for (int f = 0; f < filterVectorSize; ++f) {
    filter[f] /= total;
  }

  std::vector<float> firstPass(width*height*3, 0);

  // Gaussian blur (2 passes for rows & cols)
  for (int c = 0; c < nchannels; ++c) {
    filter1DSymmetric(firstPass, data, filter, height, width, c, /*filterRows=*/true);
    filter1DSymmetric(output, firstPass, filter, height, width, c, /*filterRows=*/false);
  }
}

static void initialSegmentation(int *output, int& nRegions, float *data, int height, int width)
{
  int const minSize = 200;
  float const threshConst = 200.f;

  // Initialise graph
  int const edgesPerVertex = 4;
  int nVerts = height*width;
  int nEdges = nVerts*edgesPerVertex;
  std::vector<float> edgeWeights(nEdges, std::numeric_limits<float>::infinity());
  int ioffs[] = {0, 1, 1, 1};
  int joffs[] = {1, 0, 1, -1};
  for (int j = 0; j < width; ++j) {
    for (int i = 0; i < height; ++i) {
      for (int d = 0; d < edgesPerVertex; ++d) {
        int di = i + ioffs[d];
        int dj = j + joffs[d];
        if (di > 0 && di < height && dj > 0 && dj < width) {
          float diff0 = data[height*width*0 + height*j + i] - data[height*width*0 + height*dj + di];
          float diff1 = data[height*width*1 + height*j + i] - data[height*width*1 + height*dj + di];
          float diff2 = data[height*width*2 + height*j + i] - data[height*width*2 + height*dj + di];
          // Cludge since the original algorithm expects uint8 image in range 0-255
          // and the threshold is based on that
          diff0 *= 255.f; diff1 *= 255.f; diff2 *= 255.f;
          float distance = sqrtf(diff0*diff0 + diff1*diff1 + diff2*diff2);
          edgeWeights[height*width*d + height*j + i] = distance;
        }
      }
    }
  }

  // Sort by edge weight
  std::vector<int> edgeIndices(edgeWeights.size());
  for (int i = 0; i < edgeIndices.size(); ++i) {
    edgeIndices[i] = i;
  }

  std::sort(edgeIndices.begin(), edgeIndices.end(), SortIndices(edgeWeights));

  std::vector<float> thresh(width*height);
  for (int i = 0; i < thresh.size(); ++i) {
    thresh[i] = threshConst;
  }

  std::vector<int> connectionMap(height * width);
  for (int i = 0; i < nVerts; ++i) {
    connectionMap[i] = i;
  }

  std::vector<int> sizeMap(nVerts, 1);
  std::vector<int> rankMap(nVerts, 0);

  for (int i = 0; i < edgeIndices.size(); ++i) {
    float weight = edgeWeights[edgeIndices[i]];
    int vertexA = edgeIndices[i] % nVerts;
    int d = edgeIndices[i] / nVerts;
    int vertexB = vertexA + height * joffs[d] + ioffs[d];

    if (weight == std::numeric_limits<float>::infinity()) {
      break;
    }

    int vertexAHead = findHead(vertexA, connectionMap);
    int vertexBHead = findHead(vertexB, connectionMap);

    if (vertexAHead != vertexBHead && !isBoundary(weight, vertexAHead, vertexBHead, thresh)) {
      // Join components
      // mexPrintf("Joining %i and %i \n", vertexAHead, vertexBHead);
      int newHead = joinRegion(vertexAHead, vertexBHead, connectionMap, sizeMap, rankMap);
      thresh[newHead] = weight + threshConst/sizeMap[newHead];
      compressPath(vertexA, connectionMap);
      compressPath(vertexB, connectionMap);
    }
  }

  // Remove regions smaller than minSize
  for (int i = 0; i < edgeIndices.size(); i++) {
    float weight = edgeWeights[edgeIndices[i]];
    int vertexA = edgeIndices[i] % nVerts;
    int d = edgeIndices[i] / nVerts;
    int vertexB = vertexA + height * joffs[d] + ioffs[d];

    if (weight == std::numeric_limits<float>::infinity()) {
      break;
    }

    int headA = findHead(vertexA, connectionMap);
    int headB = findHead(vertexB, connectionMap);
    if (headA != headB && (sizeMap[headA] < minSize || sizeMap[headB] < minSize)) {
      joinRegion(headA, headB, connectionMap, sizeMap, rankMap);
    }
  }

  // Compact indices
  std::vector<int> reindexed(nVerts, -1);

  int newIndex = 0;
  for (int i = 0; i < nVerts; ++i) {
    int head = findHead(i, connectionMap);
    if (reindexed[head] == -1) {
      reindexed[head] = newIndex;
      newIndex++;
    }
    reindexed[i] = reindexed[head];
  }

  nRegions = newIndex;

  for (int i = 0; i < nVerts; ++i) {
    output[i] = reindexed[i];
  }


}

template <typename T>
void savePPM(T& pass1, int width, int height, const char * fn, float minv=0.f, float maxv=1.f)
{
  FILE* f = fopen(fn, "w");
  fprintf(f, "P6\n%d %d\n255\n", width, height);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      float v = pass1[height*j + i];
      v = (v - minv)/(maxv - minv);
      unsigned char b = (unsigned char) std::max(0.f, std::min(255.f, 255.f * v));
      fwrite(&b,1,1,f);
      fwrite(&b,1,1,f);
      fwrite(&b,1,1,f);
    }
  }
  fclose(f);
}

template <typename T>
void saveCSV(T& pass1, int width, int height, const char * fn)
{
  FILE* f = fopen(fn, "w");
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      float v = pass1[height*j + i];
      fprintf(f, "%f", v);
      if (j != width - 1) fprintf(f, ",", v);
    }
    fprintf(f, "%\n");

  }
  fclose(f);
}

static void computeColourHistogram(std::vector<float>& histOut, float const *image, int const *segmentedImage, int height, int width, int nRegions)
{
  float const maxHistValue = 1.0f;

  // Number of histogram bins
  int const nBins = 25;
  // Size of the concatenated histograms over channels (should be 75)
  int const descriptorSize = nchannels * nBins;

  histOut.resize(nRegions * descriptorSize, 0.f);

  for (int c = 0; c < nchannels; ++c) {
    for (int i = 0; i < height * width; ++i) {
      int region = (int) segmentedImage[i];
      float value = image[c * width * height + i];

      int bin = std::min(nBins - 1, (int) ceil(value * (nBins - 1) / maxHistValue - 0.5f));
      histOut[region * descriptorSize + c * nBins + bin]++;
    }
  }

  // Normalise
  for (int r = 0; r < nRegions; ++r) {
    float sum = 0.0f;
    for(int i = 0; i < descriptorSize; ++i) {
      sum += histOut[r * descriptorSize + i];
    }
    if (sum != 0.0f) {
      for(int i = 0; i < descriptorSize; ++i) {
        histOut[r * descriptorSize + i] /= sum;
      }
    }
  }

//  // Print hists for region 18
//  for (int x=0; x < descriptorSize; ++x) {
//    printf("%f ", histOut[18 * descriptorSize + x]);
//  }printf("\n");

}

static void computeTextureHistogram(std::vector<float>& histOut, float const *image, float const *smoothImage,
                                    int const *segmentedImage, int height, int width, int nRegions)
{
  // "maximum" magnitude of the gradient (why?)
  float const maxHistValue = 0.43f;

  // Number of image derivative histograms per channel
  int const nHists = 8;
  // Number of histogram bins
  int const nBins = 10;
  // Size of the concatenated histograms over channels (should be 240)
  int const descriptorSize = nchannels * nHists * nBins;

  histOut.resize(0);
  histOut.resize(nRegions * descriptorSize, 0);

  // Layout:
  // Region 1: [channel1: [hist1,...,hist8]] [channel2: [hist1,...,hist8]] [channel3: [hist1,...,hist8]]
  // Region 2: [channel1: [hist1,...,hist8]] [channel2: [hist1,...,hist8]] [channel3: [hist1,...,hist8]]
  // ...

  // Axis-aligned gaussian derivative
  // Gaussian 2D derivative filter is separable outer product of 1D gaussian and 1D gaussian derivative
  std::vector<float> gaussian;
  std::vector<float> gaussianDeriv;
  float sum = 0.0f;
  for (int x=-4; x <= 4; ++x) {
    float g = expf(-0.5f*(x/0.8f)*(x/0.8f));
    sum += g;
    gaussian.push_back(g);
    gaussianDeriv.push_back(x);
  }
  for (int f=0; f <= gaussian.size(); ++f) {
    gaussian[f] /= sum;
    gaussianDeriv[f] = -gaussianDeriv[f] * gaussian[f] / (0.8f*0.8f);
  }

  std::vector<float> pass1(height * width * nchannels);
  std::vector<float> gradIm(height * width * nchannels);

  for (int c = 0; c < nchannels; ++c) {

    for (int histIdx = 0; histIdx < nHists/2; ++histIdx) {

      switch (histIdx) {
      case 0:
        // Y dir
        filter1D(pass1, image, gaussianDeriv, height, width, c, true);
        filter1D(gradIm, pass1, gaussian, height, width, c, false);
        break;
      case 1:
        // 45 deg
        angleGrad(gradIm, smoothImage, height, width, c, 45.f * (M_PI/180.f));
        if(c==0)saveCSV(gradIm, width, height, "/tmp/45.csv");
        break;
      case 2:
        // X dir
        filter1D(pass1, image, gaussianDeriv, height, width, c, false);
        filter1D(gradIm, pass1, gaussian, height, width, c, true);
        break;
      case 3:
        // 135 deg
        angleGrad(gradIm, smoothImage, height, width, c, 135.f * (M_PI/180.f));
        if(c==0)saveCSV(gradIm, width, height, "/tmp/135.csv");
        break;
      }

      for (int i = 0; i < height * width; ++i) {
        int region = (int) segmentedImage[i];
        float value = gradIm[i];

        // Negative component
        float nvalue = (std::max)(0.f, -value);
        int bin = (std::min)(nBins - 1, (int) ceil(nvalue * (nBins - 1) / maxHistValue - 0.5f));
        assert(bin >= 0 && bin < nBins);
        assert(region * descriptorSize + c * nHists * nBins + histIdx * nBins + bin < histOut.size());
        histOut[region * descriptorSize + c * nHists * nBins + histIdx * nBins + bin]++;

        // Positive component
        float pvalue = (std::max)(0.f, value);
        int pHistIdx = histIdx + 4;
        bin = (std::min)(nBins - 1, (int) ceil(pvalue * (nBins - 1) / maxHistValue - 0.5f));
        histOut[region * descriptorSize + c * nHists * nBins + pHistIdx * nBins + bin]++;

        //        if (histIdx == 0 && region == 18) {
        //          printf("%f -> %d\n", pvalue, bin);
        //        }
      }
    }
  }

  //  // Print hists for region 18
  //  for (int x=0; x < 240; ++x) {
  //    printf("%f ", histOut[18 * descriptorSize + x]);
  //  }printf("\n");

  // Normalise
  for (int r = 0; r < nRegions; ++r) {
    float sum = 0.0f;
    for(int i = 0; i < descriptorSize; ++i) {
      sum += histOut[r * descriptorSize + i];
    }
    if (sum != 0.0f) {
      for(int i = 0; i < descriptorSize; ++i) {
        histOut[r * descriptorSize + i] /= sum;
      }
    }
  }

  savePPM(gradIm, width, height, "/tmp/out.ppm", -0.43, 0.43);
}

void vl::selectivesearch(int *output, float const *data, int height, int width)
{
  int nRegions;
  std::vector<float> blurred(height * width * nchannels);
  std::vector<int> segmented(height * width);

  gaussianBlur(&blurred[0], data, height, width);

  initialSegmentation(&segmented[0], nRegions, &blurred[0], height, width);
  savePPM(output, width, height, "/tmp/seg.ppm", 0., nRegions);
  saveCSV(output, width, height, "/tmp/seg.csv");


  std::vector<float> histTexture;
  std::vector<float> histColour;

  computeTextureHistogram(histTexture, data, &blurred[0], &segmented[0], height, width, nRegions);
  computeColourHistogram(histColour, data, &segmented[0], height, width, nRegions);

  for (int i = 0; i < width * height; ++i) {
    output[i] = segmented[i];
  }

}
