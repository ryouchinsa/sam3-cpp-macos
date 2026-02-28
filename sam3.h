#ifndef SAM3_CPP_H_
#define SAM3_CPP_H_

#include <onnxruntime_cxx_api.h>
#include <tokenizers_cpp.h>
#include <opencv2/core.hpp>
#include <list>
#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include "util.h"

using tokenizers::Tokenizer;

class Sam3 {
  std::unique_ptr<Ort::Session> visionEncoder, textEncoder, geometryEncoder, decoder;
  std::unique_ptr<Tokenizer> tokenizer;
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
  Ort::SessionOptions sessionOptions;
  Ort::RunOptions runOptionsEncoder;
  Ort::MemoryInfo memoryInfo{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::vector<int64_t> inputShapeVision;
  std::vector<int64_t> outputShapeVision[4];
  std::vector<float> outputVision[4];
  std::vector<int64_t> inputShapeText[2];
  std::vector<int64_t> outputShapeText[2];
  std::vector<float> outputText0;
  std::vector<uint8_t> outputText1;
  std::vector<int64_t> outputShapeBoxes[2];
  std::vector<float> outputBoxes0;
  std::vector<uint8_t> outputBoxes1;
  std::vector<int64_t> outputShapeDecoder[4];
  std::vector<float> outputDecoder[4];
  bool loadingModel = false;
  bool preprocessing = false;
  bool terminating = false;
 public:
  Sam3();
  ~Sam3();
  bool clearLoadModel();
  void terminatePreprocessing();
  bool loadModel(const std::string& visionPath, const std::string& textPath, const std::string& geometryPath, const std::string& decoderPath, const std::string& tokenizerPath, int threadsNumber, const std::string device);
  void loadingStart();
  void loadingEnd();
  cv::Size getInputSize();
  bool preprocessImage(const cv::Mat& image);
  void preprocessingStart();
  void preprocessingEnd();
  bool encodeText(const std::string& text);
  bool encodeBoxes(const std::vector<cv::Rect2f> &rects, const std::vector<int> &labels);
  std::tuple<std::vector<cv::Mat>, std::vector<int>> decode(float threshold, const cv::Size &imageSize, bool skipDecode);
  std::tuple<std::vector<cv::Mat>, std::vector<int>> changeThreshold(float threshold, const cv::Size &imageSize);
  bool isDecoderEmpty();
};

#endif
