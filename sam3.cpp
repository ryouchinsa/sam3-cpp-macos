#include "sam3.h"
#include <opencv2/opencv.hpp>

Sam3::Sam3(){}
Sam3::~Sam3(){
  if(loadingModel){
    return;
  }
  if(preprocessing){
    return;
  }
  clearLoadModel();
}

bool Sam3::clearLoadModel(){
  try{
    Ort::Session* v = visionEncoder.release();
    Ort::Session* t = textEncoder.release();
    Ort::Session* g = geometryEncoder.release();
    Ort::Session* d = decoder.release();
    delete v;
    delete t;
    delete g;
    delete d;
    inputShapeVision.resize(0);
    for(int i = 0; i < 4; i++){
      outputShapeVision[i].resize(0);
      outputVision[i].resize(0);
    }
    for(int i = 0; i < 2; i++){
      inputShapeText[i].resize(0);
      outputShapeText[i].resize(0);
    }
    outputText0.resize(0);
    outputText1.resize(0);
    for(int i = 0; i < 2; i++){
      outputShapeBoxes[i].resize(0);
    }
    outputBoxes0.resize(0);
    outputBoxes1.resize(0);
    for(int i = 0; i < 4; i++){
      outputShapeDecoder[i].resize(0);
      outputDecoder[i].resize(0);
    }
  }catch(Ort::Exception& e){
    return false;
  }
  return true;
}

void Sam3::terminatePreprocessing(){
  runOptionsEncoder.SetTerminate();
  terminating = true;
}

bool Sam3::loadModel(const std::string& visionPath, const std::string& textPath, const std::string& geometryPath, const std::string& decoderPath, const std::string& tokenizerPath, int threadsNumber, const std::string device){
  try{
    loadingStart();
    if(!clearLoadModel()){
      loadingEnd();
      return false;
    }
    if(!modelExists(visionPath) || !modelExists(textPath) || !modelExists(geometryPath) || !modelExists(decoderPath) || !modelExists(tokenizerPath)){
      loadingEnd();
      return false;
    }
    sessionOptions.SetIntraOpNumThreads(threadsNumber);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if(device.substr(0, 5) == "cuda:"){
      int gpuDeviceId = std::stoi(device.substr(5));
      OrtCUDAProviderOptions options;
      options.device_id = gpuDeviceId;
      sessionOptions.AppendExecutionProvider_CUDA(options);
    }
    visionEncoder = std::make_unique<Ort::Session>(env, visionPath.c_str(), sessionOptions);
    textEncoder = std::make_unique<Ort::Session>(env, textPath.c_str(), sessionOptions);
    geometryEncoder = std::make_unique<Ort::Session>(env, geometryPath.c_str(), sessionOptions);
    decoder = std::make_unique<Ort::Session>(env, decoderPath.c_str(), sessionOptions);
    auto blob = LoadBytesFromFile(tokenizerPath.c_str());
    tokenizer = Tokenizer::FromBlobJSON(blob);
    
    inputShapeVision = visionEncoder->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    inputShapeVision[0] = 1;
    outputShapeVision[3] = visionEncoder->GetOutputTypeInfo(3).GetTensorTypeAndShapeInfo().GetShape();
    outputShapeVision[3][0] = 1;
    outputShapeVision[2] = outputShapeVision[3];
    outputShapeVision[1] = outputShapeVision[3];
    outputShapeVision[1][2] = outputShapeVision[1][3] = outputShapeVision[1][2] * 2;
    outputShapeVision[0] = outputShapeVision[3];
    outputShapeVision[0][2] = outputShapeVision[0][3] = outputShapeVision[0][2] * 4;
    
    inputShapeText[0] = textEncoder->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    inputShapeText[1] = textEncoder->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
    outputShapeText[0] = textEncoder->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    outputShapeText[1] = textEncoder->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
    inputShapeText[0][0] = 1;
    inputShapeText[1][0] = 1;
    outputShapeText[0][0] = 1;
    outputShapeText[1][0] = 1;
  }catch(Ort::Exception& e){
    std::cout << e.what() << std::endl;
    loadingEnd();
    return false;
  }
  if(terminating){
    loadingEnd();
    return false;
  }
  loadingEnd();
  return true;
}

void Sam3::loadingStart(){
  loadingModel = true;
}

void Sam3::loadingEnd(){
  loadingModel = false;
  terminating = false;
}

cv::Size Sam3::getInputSize(){
  return cv::Size((int)inputShapeVision[3], (int)inputShapeVision[2]);
}

bool Sam3::preprocessImage(const cv::Mat& image){
  try{
    preprocessingStart();
    if(image.size() != cv::Size((int)inputShapeVision[3], (int)inputShapeVision[2])){
      preprocessingEnd();
      return false;
    }
    if(image.channels() != 3){
      preprocessingEnd();
      return false;
    }
    std::vector<float> inputTensorValuesFloat(getShapeSize(inputShapeVision));
    for(int i = 0; i < inputShapeVision[2]; i++){
      for(int j = 0; j < inputShapeVision[3]; j++){
        int64_t pos = i * inputShapeVision[3] + j;
        int64_t size = inputShapeVision[2] * inputShapeVision[3];
        inputTensorValuesFloat[pos + size * 0] = image.at<cv::Vec3b>(i, j)[2] / 127.5 - 1.0;
        inputTensorValuesFloat[pos + size * 1] = image.at<cv::Vec3b>(i, j)[1] / 127.5 - 1.0;
        inputTensorValuesFloat[pos + size * 2] = image.at<cv::Vec3b>(i, j)[0] / 127.5 - 1.0;
      }
    }
    auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValuesFloat.data(), inputTensorValuesFloat.size(), inputShapeVision.data(), inputShapeVision.size());
    std::vector<Ort::Value> outputTensors;
    for(int i = 0; i < 4; i++){
      outputVision[i] = std::vector<float>(getShapeSize(outputShapeVision[i]));
      outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputVision[i].data(), outputVision[i].size(), outputShapeVision[i].data(), outputShapeVision[i].size()));
    }
    if(terminating){
      preprocessingEnd();
      return false;
    }
    runOptionsEncoder.UnsetTerminate();
    std::vector<const char*> inputNames = getInputNames(visionEncoder);
    std::vector<const char*> outputNames = getOutputNames(visionEncoder);
    visionEncoder->Run(runOptionsEncoder, inputNames.data(), &inputTensor, 1, outputNames.data(), outputTensors.data(), outputTensors.size());
    for (size_t i = 0; i < inputNames.size(); ++i) {
      delete [] inputNames[i];
    }
    for (size_t i = 0; i < outputNames.size(); ++i) {
      delete [] outputNames[i];
    }
  }catch(Ort::Exception& e){
    std::cout << e.what() << std::endl;
    preprocessingEnd();
    return false;
  }
  preprocessingEnd();
  return true;
}

void Sam3::preprocessingStart(){
  preprocessing = true;
}

void Sam3::preprocessingEnd(){
  preprocessing = false;
  terminating = false;
}

bool Sam3::encodeText(const std::string& text){
  try{
    preprocessingStart();
    std::vector<int64_t> inputTensorValues[2];
    for(int i = 0; i < 2; i++){
      inputTensorValues[i].resize(getShapeSize(inputShapeText[i]));
    }
    if(text.length() > 0){
      std::vector<int> ids = tokenizer->Encode(text);
      ids.insert(ids.begin(), 49406);
      ids.push_back(49407);
      for(int i = 0; i < inputTensorValues[0].size(); i++){
        if(i < ids.size()){
          inputTensorValues[0][i] = ids[i];
          inputTensorValues[1][i] = 1;
        }else{
          inputTensorValues[0][i] = 49407;
          inputTensorValues[1][i] = 0;
        }
      }
    }else{
      for(int i = 0; i < inputTensorValues[0].size(); i++){
        inputTensorValues[0][i] = 49407;
        if(i == 0){
          inputTensorValues[1][i] = 1;
        }else{
          inputTensorValues[1][i] = 0;
        }
      }
    }
    std::vector<Ort::Value> inputTensors;
    for(int i = 0; i < 2; i++){
      inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, inputTensorValues[i].data(), inputTensorValues[i].size(), inputShapeText[i].data(), inputShapeText[i].size()));
    }
    outputText0 = std::vector<float>(getShapeSize(outputShapeText[0]));
    outputText1 = std::vector<uint8_t>(getShapeSize(outputShapeText[1]));
    uint8_t *ptrOutputText1 = outputText1.data();
    bool *ptrOutputText1Bool = reinterpret_cast<bool*>(ptrOutputText1);
    std::vector<Ort::Value> outputTensors;
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputText0.data(), outputText0.size(), outputShapeText[0].data(), outputShapeText[0].size()));
    outputTensors.push_back(Ort::Value::CreateTensor<bool>(memoryInfo, ptrOutputText1Bool, outputText1.size(), outputShapeText[1].data(), outputShapeText[1].size()));
    if(terminating){
      preprocessingEnd();
      return false;
    }
    runOptionsEncoder.UnsetTerminate();
    std::vector<const char*> inputNames = getInputNames(textEncoder);
    std::vector<const char*> outputNames = getOutputNames(textEncoder);
    textEncoder->Run(runOptionsEncoder, inputNames.data(), inputTensors.data(), inputTensors.size(), outputNames.data(), outputTensors.data(), outputTensors.size());
    for (size_t i = 0; i < inputNames.size(); ++i) {
      delete [] inputNames[i];
    }
    for (size_t i = 0; i < outputNames.size(); ++i) {
      delete [] outputNames[i];
    }
  }catch(Ort::Exception& e){
    std::cout << e.what() << std::endl;
    preprocessingEnd();
    return false;
  }
  preprocessingEnd();
  return true;
}

bool Sam3::encodeBoxes(const std::vector<cv::Rect2f> &rects, const std::vector<int> &labels){
  try{
    if(rects.size() == 0){
      outputBoxes0.resize(0);
      outputBoxes1.resize(0);
      return true;
    }
    preprocessingStart();
    std::vector<float> inputTensorValues0;
    std::vector<int64_t> inputTensorValues1;
    for(int i = 0; i < rects.size(); i++){
      inputTensorValues0.push_back(rects[i].x);
      inputTensorValues0.push_back(rects[i].y);
      inputTensorValues0.push_back(rects[i].width);
      inputTensorValues0.push_back(rects[i].height);
      inputTensorValues1.push_back(labels[i]);
    }
    std::vector<int64_t> inputShape0, inputShape1;
    inputShape0.push_back(1);
    inputShape0.push_back(rects.size());
    inputShape0.push_back(4);
    inputShape1.push_back(1);
    inputShape1.push_back(rects.size());
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues0.data(), inputTensorValues0.size(), inputShape0.data(), inputShape0.size()));
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, inputTensorValues1.data(), inputTensorValues1.size(), inputShape1.data(), inputShape1.size()));
    for(int i = 2; i < 4; i++){
      inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputVision[i].data(), outputVision[i].size(), outputShapeVision[i].data(), outputShapeVision[i].size()));
    }
    for(int i = 0; i < 2; i++){
      outputShapeBoxes[i] = outputShapeText[i];
    }
    outputShapeBoxes[0][1] = outputShapeBoxes[1][1] = rects.size() + 1;
    outputBoxes0 = std::vector<float>(getShapeSize(outputShapeBoxes[0]));
    outputBoxes1 = std::vector<uint8_t>(getShapeSize(outputShapeBoxes[1]));
    uint8_t *ptrOutputBoxes1 = outputBoxes1.data();
    bool *ptrOutputBoxes1Bool = reinterpret_cast<bool*>(ptrOutputBoxes1);
    std::vector<Ort::Value> outputTensors;
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputBoxes0.data(), outputBoxes0.size(), outputShapeBoxes[0].data(), outputShapeBoxes[0].size()));
    outputTensors.push_back(Ort::Value::CreateTensor<bool>(memoryInfo, ptrOutputBoxes1Bool, outputBoxes1.size(), outputShapeBoxes[1].data(), outputShapeBoxes[1].size()));
    if(terminating){
      preprocessingEnd();
      return false;
    }
    runOptionsEncoder.UnsetTerminate();
    std::vector<const char*> inputNames = getInputNames(geometryEncoder);
    std::vector<const char*> outputNames = getOutputNames(geometryEncoder);
    geometryEncoder->Run(runOptionsEncoder, inputNames.data(), inputTensors.data(), inputTensors.size(), outputNames.data(), outputTensors.data(), outputTensors.size());
    for (size_t i = 0; i < inputNames.size(); ++i) {
      delete [] inputNames[i];
    }
    for (size_t i = 0; i < outputNames.size(); ++i) {
      delete [] outputNames[i];
    }
  }catch(Ort::Exception& e){
    std::cout << e.what() << std::endl;
    preprocessingEnd();
    return false;
  }
  preprocessingEnd();
  return true;
}

std::tuple<std::vector<cv::Mat>, std::vector<int>> Sam3::decode(float threshold, const cv::Size &imageSize, bool skipDecode){
  if(skipDecode){
    return changeThreshold(threshold, imageSize);
  }
  preprocessingStart();
  std::vector<cv::Mat> masks;
  std::vector<int> boxes;
  try{
    std::vector<Ort::Value> inputTensors;
    for(int i = 0; i < 4; i++){
      inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputVision[i].data(), outputVision[i].size(), outputShapeVision[i].data(), outputShapeVision[i].size()));
    }
    std::vector<float> outputTextBoxes0 = outputText0;
    std::vector<uint8_t> outputTextBoxes1 = outputText1;
    std::vector<int64_t> outputShapeTextBoxes[2];
    for(int i = 0; i < 2; i++){
      outputShapeTextBoxes[i] = outputShapeText[i];
    }
    if(!outputBoxes0.empty()){
      outputTextBoxes0.insert(outputTextBoxes0.end(), outputBoxes0.begin(), outputBoxes0.end());
      outputTextBoxes1.insert(outputTextBoxes1.end(), outputBoxes1.begin(), outputBoxes1.end());
      for(int i = 0; i < 2; i++){
        outputShapeTextBoxes[i][1] += outputShapeBoxes[i][1];
      }
    }
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTextBoxes0.data(), outputTextBoxes0.size(), outputShapeTextBoxes[0].data(), outputShapeTextBoxes[0].size()));
    uint8_t *ptrOutputTextBoxes1 = outputTextBoxes1.data();
    bool *ptrOutputTextBoxes1Bool = reinterpret_cast<bool*>(ptrOutputTextBoxes1);
    inputTensors.push_back(Ort::Value::CreateTensor<bool>(memoryInfo, ptrOutputTextBoxes1Bool, outputTextBoxes1.size(), outputShapeTextBoxes[1].data(), outputShapeTextBoxes[1].size()));
    if(terminating){
      preprocessingEnd();
      return std::make_tuple(masks, boxes);
    }
    runOptionsEncoder.UnsetTerminate();
    std::vector<const char*> inputNames = getInputNames(decoder);
    std::vector<const char*> outputNames = getOutputNames(decoder);
    auto outputTensors = decoder->Run(runOptionsEncoder, inputNames.data(), inputTensors.data(), inputTensors.size(), outputNames.data(), outputNames.size());
    for (size_t i = 0; i < inputNames.size(); ++i) {
      delete [] inputNames[i];
    }
    for (size_t i = 0; i < outputNames.size(); ++i) {
      delete [] outputNames[i];
    }
    for(int i = 0; i < 4; i++){
      auto values = outputTensors[i].GetTensorMutableData<float>();
      outputShapeDecoder[i] = outputTensors[i].GetTensorTypeAndShapeInfo().GetShape();
      outputDecoder[i].assign(values, values + getShapeSize(outputShapeDecoder[i]));
    }
  }catch(Ort::Exception& e){
    std::cout << e.what() << std::endl;
    preprocessingEnd();
    return std::make_tuple(masks, boxes);
  }
  preprocessingEnd();
  return changeThreshold(threshold, imageSize);
}

std::tuple<std::vector<cv::Mat>, std::vector<int>> Sam3::changeThreshold(float threshold, const cv::Size &imageSize){
  preprocessingStart();
  std::vector<cv::Mat> masks;
  std::vector<int> boxes;
  float presence_logits = outputDecoder[3][0];
  float presence_score = 1 / (1 + exp(-presence_logits));
  std::vector<bool> keep(outputDecoder[2].size());
  std::vector<float> scores(outputDecoder[2].size());
  int count = 0;
  for(int i = 0; i < outputDecoder[2].size(); i++){
    scores[i] = (1 / (1 + exp(-outputDecoder[2][i]))) * presence_score;
    if(scores[i] > threshold){
      keep[i] = true;
      count++;
    }else{
      keep[i] = false;
    }
  }
  std::vector<int> sort_ids = sort_indexes(scores);
  for(int s = 0; s < sort_ids.size(); s++){
    int k = sort_ids[s];
    if(!keep[k]){
      continue;
    }
    std::vector<int> box;
    for (int i = 0; i < 4; i++) {
      float value = outputDecoder[1][k * 4 + i];
      if(i % 2 == 0){
        value *= imageSize.width;
      }else{
        value *= imageSize.height;
      }
      box.push_back(value);
    }
    for (int i = 0; i < 4; i++) {
      boxes.push_back(box[i]);
    }
    cv::Mat maskf((int)outputShapeDecoder[0][2], (int)outputShapeDecoder[0][3], CV_32F, cv::Scalar(0));
    for (int i = 0; i < maskf.rows; i++) {
      for (int j = 0; j < maskf.cols; j++) {
        maskf.at<float>(i, j) = outputDecoder[0][k * maskf.rows * maskf.cols + i * maskf.cols + j];
      }
    }
    cv::resize(maskf, maskf, imageSize, 0, 0, cv::INTER_LINEAR);
    cv::Mat mask(imageSize.height, imageSize.width, CV_8UC1, cv::Scalar(0));
    mask.setTo(255, maskf > 0);
    masks.push_back(mask);
  }
  preprocessingEnd();
  return std::make_tuple(masks, boxes);
}

bool Sam3::isDecoderEmpty(){
  if(outputDecoder[0].size() == 0){
    return true;
  }
  return false;
}

