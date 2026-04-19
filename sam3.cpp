#include "sam3.h"
#include <opencv2/opencv.hpp>
#include <future>

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
    Ort::Session* d = decoder.release();
    delete v;
    delete t;
    delete d;
    inputTensorValuesFloat.resize(0);
    inputShapeVision.resize(0);
    for(int i = 0; i < 4; i++){
      outputShapeVision[i].resize(0);
      outputVision[i].resize(0);
    }
    clearVisionBatch();
    for(int i = 0; i < 2; i++){
      inputShapeText[i].resize(0);
      outputShapeText[i].resize(0);
    }
    outputText0.resize(0);
    outputText1.resize(0);
    clearDecoder();
  }catch(Ort::Exception& e){
    return false;
  }
  return true;
}

void Sam3::clearVisionBatch(){
  for(int i = 0; i < 4; i++){
    outputShapeVisionBatch[i].resize(0);
    outputVisionBatch[i].resize(0);
  }
}

void Sam3::clearDecoder(){
  for(int i = 0; i < 4; i++){
    outputShapeDecoder[i].resize(0);
    outputDecoder[i].resize(0);
  }
}

bool Sam3::isDecoderEmpty(){
  if(outputDecoder[0].size() == 0){
    return true;
  }
  return false;
}

void Sam3::terminatePreprocessing(){
  runOptionsEncoder.SetTerminate();
  terminating = true;
}

bool Sam3::loadModel(const std::string& visionPath, const std::string& textPath, const std::string& decoderPath, const std::string& tokenizerPath, int threadsNumber, const std::string device){
  try{
    loadingStart();
    if(!clearLoadModel()){
      loadingEnd();
      return false;
    }
    if(!modelExists(visionPath) || !modelExists(textPath) || !modelExists(decoderPath) || !modelExists(tokenizerPath)){
      loadingEnd();
      return false;
    }

    // Use global thread pool like Python's onnxruntime does
    Ort::ThreadingOptions threadingOptions;
    threadingOptions.SetGlobalIntraOpNumThreads(threadsNumber);
    threadingOptions.SetGlobalInterOpNumThreads(threadsNumber);

    // Replace the Env — must be done before session creation
    env = Ort::Env(threadingOptions, ORT_LOGGING_LEVEL_WARNING, "test");

    sessionOptions.SetIntraOpNumThreads(threadsNumber);
    sessionOptions.SetInterOpNumThreads(threadsNumber);  // <-- this was missing
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Disable per-session thread spinning — let global pool handle it
    sessionOptions.AddConfigEntry("session.intra_op.allow_spinning", "0");

    // Enable memory pattern optimization
    sessionOptions.EnableMemPattern();
    sessionOptions.EnableCpuMemArena();

    if(device.substr(0, 5) == "cuda:"){
      int gpuDeviceId = std::stoi(device.substr(5));
      OrtCUDAProviderOptions options;
      options.device_id = gpuDeviceId;
      sessionOptions.AppendExecutionProvider_CUDA(options);
    }

    // Replace the three make_unique lines in loadModel() with:
    auto futureVision = std::async(std::launch::async, [&](){
      return std::make_unique<Ort::Session>(env, visionPath.c_str(), sessionOptions);
    });
    auto futureText = std::async(std::launch::async, [&](){
      return std::make_unique<Ort::Session>(env, textPath.c_str(), sessionOptions);
    });
    auto futureDecoder = std::async(std::launch::async, [&](){
      return std::make_unique<Ort::Session>(env, decoderPath.c_str(), sessionOptions);
    });
    auto futureTokenizer = std::async(std::launch::async, [&](){
      auto blob = LoadBytesFromFile(tokenizerPath.c_str());
      return Tokenizer::FromBlobJSON(blob);
    });
    visionEncoder = futureVision.get();
    textEncoder   = futureText.get();
    decoder       = futureDecoder.get();
    tokenizer     = futureTokenizer.get();

    auto cacheIONames = [](Ort::Session* sess,
                           std::vector<std::string>& inNames,  std::vector<const char*>& inPtrs,
                           std::vector<std::string>& outNames, std::vector<const char*>& outPtrs){
      Ort::AllocatorWithDefaultOptions alloc;

      inNames.clear();
      for(size_t i = 0; i < sess->GetInputCount(); i++)
        inNames.push_back(sess->GetInputNameAllocated(i, alloc).get());

      outNames.clear();
      for(size_t i = 0; i < sess->GetOutputCount(); i++)
        outNames.push_back(sess->GetOutputNameAllocated(i, alloc).get());

      // Only build pointer vectors AFTER all strings are final — no more reallocation
      inPtrs.clear();
      for(auto& s : inNames)  inPtrs.push_back(s.c_str());

      outPtrs.clear();
      for(auto& s : outNames) outPtrs.push_back(s.c_str());
    };
    cacheIONames(visionEncoder.get(), cachedInputNamesVision, ptrInputNamesVision,
                                      cachedOutputNamesVision, ptrOutputNamesVision);
    cacheIONames(textEncoder.get(),   cachedInputNamesText,   ptrInputNamesText,
                                      cachedOutputNamesText,   ptrOutputNamesText);
    cacheIONames(decoder.get(),       cachedInputNamesDecoder, ptrInputNamesDecoder,
                                      cachedOutputNamesDecoder, ptrOutputNamesDecoder);
    
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

    inputTensorValuesFloat.assign(getShapeSize(inputShapeVision), 0.0f);
    for(int i = 0; i < 4; i++){
      outputVision[i].assign(getShapeSize(outputShapeVision[i]), 0.0f);
    }
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

    // FAST: vectorized OpenCV ops matching Python's (img / 127.5 - 1.0).transpose(2,0,1)
    cv::Mat imageFloat;
    image.convertTo(imageFloat, CV_32F, 1.0 / 127.5, -1.0); // bgr float, normalized

    // Split into B, G, R planes and reorder to R, G, B (CHW layout)
    std::vector<cv::Mat> channels(3);
    cv::split(imageFloat, channels);  // channels[0]=B, [1]=G, [2]=R

    int64_t planeSize = inputShapeVision[2] * inputShapeVision[3];
    // Copy R, G, B into CHW tensor (matching Python's channel order)
    std::memcpy(inputTensorValuesFloat.data() + 0 * planeSize,
                channels[2].ptr<float>(), planeSize * sizeof(float)); // R
    std::memcpy(inputTensorValuesFloat.data() + 1 * planeSize,
                channels[1].ptr<float>(), planeSize * sizeof(float)); // G
    std::memcpy(inputTensorValuesFloat.data() + 2 * planeSize,
                channels[0].ptr<float>(), planeSize * sizeof(float)); // B

    auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValuesFloat.data(), inputTensorValuesFloat.size(), inputShapeVision.data(), inputShapeVision.size());
    std::vector<Ort::Value> outputTensors;
    for(int i = 0; i < 4; i++){
      outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputVision[i].data(), outputVision[i].size(),
        outputShapeVision[i].data(), outputShapeVision[i].size()));
    }
    if(terminating){
      preprocessingEnd();
      return false;
    }
    runOptionsEncoder.UnsetTerminate();
    visionEncoder->Run(runOptionsEncoder,
      ptrInputNamesVision.data(),  &inputTensor, 1,
      ptrOutputNamesVision.data(), outputTensors.data(), outputTensors.size());
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

bool Sam3::encodeText(const std::vector<std::string> &text_list){
  try{
    preprocessingStart();
    int batchSize = (int)text_list.size();
    if(batchSize == 0){
      batchSize = 1;
    }
    inputShapeText[0][0] = batchSize;
    inputShapeText[1][0] = batchSize;
    outputShapeText[0][0] = batchSize;
    outputShapeText[1][0] = batchSize;
    std::vector<int64_t> inputTensorValues[2];
    for(int i = 0; i < 2; i++){
      inputTensorValues[i].resize(getShapeSize(inputShapeText[i]));
    }
    for(int b = 0; b < batchSize; b++){
      int offset = b * (int)inputShapeText[0][1];
      std::string text = "";
      if(b < text_list.size()){
        text = text_list[b];
      }
      if(text.length() > 0){
        std::vector<int> ids = tokenizer->Encode(text);
        ids.insert(ids.begin(), 49406);
        ids.push_back(49407);
        for(int i = 0; i < inputShapeText[0][1]; i++){
          if(i < ids.size()){
            inputTensorValues[0][i + offset] = ids[i];
            inputTensorValues[1][i + offset] = 1;
          }else{
            inputTensorValues[0][i + offset] = 49407;
            inputTensorValues[1][i + offset] = 0;
          }
        }
      }
      // if(text.length() > 0){
      //   std::vector<int> ids_raw = tokenizer->Encode(text);

      //   // Write directly into inputTensorValues without building intermediate ids vector
      //   int slot = 0;
      //   auto write = [&](int id, int mask_val){
      //     if(slot < inputShapeText[0][1]){
      //       inputTensorValues[0][slot + offset] = id;
      //       inputTensorValues[1][slot + offset] = mask_val;
      //       slot++;
      //     }
      //   };

      //   write(49406, 1);                          // BOS token
      //   for(int id : ids_raw) write(id, 1);       // token ids
      //   write(49407, 1);                          // EOS token

      //   // Pad remainder
      //   while(slot < inputShapeText[0][1]){
      //     inputTensorValues[0][slot + offset] = 49407;
      //     inputTensorValues[1][slot + offset] = 0;
      //     slot++;
      //   }
      // }
      else{
        for(int i = 0; i < inputShapeText[0][1]; i++){
          inputTensorValues[0][i + offset] = 49407;
          if(i == 0){
            inputTensorValues[1][i + offset] = 1;
          }else{
            inputTensorValues[1][i + offset] = 0;
          }
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
    textEncoder->Run(runOptionsEncoder,
      ptrInputNamesText.data(),  inputTensors.data(), inputTensors.size(),
      ptrOutputNamesText.data(), outputTensors.data(), outputTensors.size());
  }catch(Ort::Exception& e){
    std::cout << e.what() << std::endl;
    preprocessingEnd();
    return false;
  }
  preprocessingEnd();
  return true;
}

void Sam3::alignTextsAndBoxes(std::vector<std::string> *text_list, std::vector<std::vector<cv::Rect2f>> *rects_list, std::vector<std::vector<int>> *labels_list){
  int textNum = (int)(*text_list).size();
  int boxNum = (int)(*rects_list).size();
  int batchSizeText = textNum;
  if(batchSizeText == 0){
    batchSizeText = 1;
  }
  int batchSizeBoxes = boxNum;
  if(batchSizeText == batchSizeBoxes){
    return;
  }
  if(batchSizeBoxes > batchSizeText){
    int addNum = batchSizeBoxes - textNum;
    for(int i = 0; i < addNum; i++){
      (*text_list).push_back("");
    }
    return;
  }
  int addNum = batchSizeText - batchSizeBoxes;
  for(int i = 0; i < addNum; i++){
    std::vector<cv::Rect2f> rects;
    std::vector<int> labels;
    (*rects_list).push_back(rects);
    (*labels_list).push_back(labels);
  }
}

void Sam3::setOutputVisionToInputTensors(int batchSize, std::vector<Ort::Value> *inputTensors){
  if(batchSize == 1){
    clearVisionBatch();
    for(int i = 0; i < 4; i++){
      (*inputTensors).push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputVision[i].data(), outputVision[i].size(), outputShapeVision[i] .data(), outputShapeVision[i] .size()));
    }
    return;
  }
  std::vector<int64_t> shape = outputShapeVisionBatch[0];
  if(shape.size() > 0 && shape[0] == batchSize){
    return;
  }
  clearVisionBatch();
  for(int i = 0; i < 4; i++){
    for(int b = 0; b < batchSize; b++){
      outputVisionBatch[i].insert(outputVisionBatch[i].end(), outputVision[i].begin(), outputVision[i].end());
    }
    outputShapeVisionBatch[i] = outputShapeVision[i];
    outputShapeVisionBatch[i][0] = batchSize;
    (*inputTensors).push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputVisionBatch[i].data(), outputVisionBatch[i].size(), outputShapeVisionBatch[i] .data(), outputShapeVisionBatch[i] .size()));
  }
}

std::tuple<std::vector<cv::Mat>, std::vector<int>> Sam3::decode(const std::vector<std::vector<cv::Rect2f>> &rects_list, const std::vector<std::vector<int>> &labels_list, float threshold, const cv::Size &imageSize, bool skipDecode){
  if(skipDecode){
    return changeThreshold(threshold, imageSize);
  }
  std::chrono::steady_clock::time_point begin, end;
  begin = std::chrono::steady_clock::now();
  preprocessingStart();
  clearDecoder();
  std::vector<cv::Mat> masks;
  std::vector<int> boxes;
  try{
    int batchSize = (int)inputShapeText[0][0];
    std::vector<Ort::Value> inputTensors;
    setOutputVisionToInputTensors(batchSize, &inputTensors);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputText0.data(), outputText0.size(), outputShapeText[0].data(), outputShapeText[0].size()));
    uint8_t *ptrOutputText1 = outputText1.data();
    bool *ptrOutputText1Bool = reinterpret_cast<bool*>(ptrOutputText1);
    inputTensors.push_back(Ort::Value::CreateTensor<bool>(memoryInfo, ptrOutputText1Bool, outputText1.size(), outputShapeText[1].data(), outputShapeText[1].size()));

    int boxNumMax = 0;
    for(int b = 0; b < batchSize; b++){
      std::vector<cv::Rect2f> rects = rects_list[b];
      if(rects.size() > boxNumMax){
        boxNumMax = (int)rects.size();
      }
    }
    if(boxNumMax == 0){
      boxNumMax = 1;
    }
    std::vector<float> inputTensorValues0;
    std::vector<int64_t> inputTensorValues1;
    for(int b = 0; b < batchSize; b++){
      const std::vector<cv::Rect2f>& rects = rects_list[b];
      const std::vector<int>& labels = labels_list[b];
      for(int i = 0; i < rects.size(); i++){
        inputTensorValues0.push_back(rects[i].x);
        inputTensorValues0.push_back(rects[i].y);
        inputTensorValues0.push_back(rects[i].width);
        inputTensorValues0.push_back(rects[i].height);
        inputTensorValues1.push_back(labels[i]);
      }
      for(int i = (int)rects.size(); i < boxNumMax; i++){
        inputTensorValues0.push_back(0);
        inputTensorValues0.push_back(0);
        inputTensorValues0.push_back(0);
        inputTensorValues0.push_back(0);
        inputTensorValues1.push_back(-10);
      }
    }
    std::vector<int64_t> inputShape0, inputShape1;
    inputShape0.push_back(batchSize);
    inputShape0.push_back(boxNumMax);
    inputShape0.push_back(4);
    inputShape1.push_back(batchSize);
    inputShape1.push_back(boxNumMax);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues0.data(), inputTensorValues0.size(), inputShape0.data(), inputShape0.size()));
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, inputTensorValues1.data(), inputTensorValues1.size(), inputShape1.data(), inputShape1.size()));

    if(terminating){
      preprocessingEnd();
      return std::make_tuple(masks, boxes);
    }
    runOptionsEncoder.UnsetTerminate();

    // // Allocate output buffers based on actual batch size (done just before this)
    // for(int i = 0; i < 4; i++){
    //   outputShapeDecoder[i] = decoder->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    //   outputShapeDecoder[i][0] = batchSize;
    // }
    // outputShapeDecoder[0][2] = outputShapeDecoder[0][3] = outputShapeVision[0][2];
    // outputShapeDecoder[0][1] = outputShapeDecoder[1][1];
    // outputShapeDecoder[2][1] = outputShapeDecoder[1][1];
    // outputShapeDecoder[3][1] = 1;
    // for(int i = 0; i < 4; i++){
    //   printShape(outputShapeDecoder[i]);
    //   outputDecoder[i].resize(getShapeSize(outputShapeDecoder[i]));
    // }

    // // Pre-bind output tensors so ORT writes directly into outputDecoder — no post-run copy
    // std::vector<Ort::Value> decoderOutputTensors;
    // for(int i = 0; i < 4; i++){
    //   decoderOutputTensors.push_back(Ort::Value::CreateTensor<float>(
    //     memoryInfo, outputDecoder[i].data(), outputDecoder[i].size(),
    //     outputShapeDecoder[i].data(), outputShapeDecoder[i].size()));
    // }

    // decoder->Run(runOptionsEncoder,
    //   ptrInputNamesDecoder.data(),  inputTensors.data(),          inputTensors.size(),
    //   ptrOutputNamesDecoder.data(), decoderOutputTensors.data(),  decoderOutputTensors.size());

    auto outputTensors = decoder->Run(runOptionsEncoder,
      ptrInputNamesDecoder.data(), inputTensors.data(), inputTensors.size(),
      ptrOutputNamesDecoder.data(), ptrOutputNamesDecoder.size());
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
  end = std::chrono::steady_clock::now();
  std::cout << "decode sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  return changeThreshold(threshold, imageSize);
}

std::tuple<std::vector<cv::Mat>, std::vector<int>> Sam3::changeThreshold(float threshold, const cv::Size &imageSize){
  std::chrono::steady_clock::time_point begin, end;
  begin = std::chrono::steady_clock::now();
  preprocessingStart();
  std::vector<cv::Mat> masks;
  std::vector<int> boxes;
  int batchSize = (int)outputShapeDecoder[0][0];
  int scoreSize = (int)outputShapeDecoder[2][1];
  int boxSize = (int)(outputShapeDecoder[1][1] * outputShapeDecoder[1][2]);
  int maskSize = (int)(outputShapeDecoder[0][1] * outputShapeDecoder[0][2] * outputShapeDecoder[0][3]);
  for(int b = 0; b < batchSize; b++){
    float presence_logits = outputDecoder[3][b];
    float presence_score = 1 / (1 + exp(-presence_logits));
    std::vector<bool> keep(scoreSize);
    std::vector<float> scores(scoreSize);
    int count = 0;
    for(int i = 0; i < scoreSize; i++){
      scores[i] = (1 / (1 + exp(-outputDecoder[2][i + b * scoreSize]))) * presence_score;
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
        float value = outputDecoder[1][k * 4 + i + b * boxSize];
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
      // REPLACE the maskf construction + inner loop:
      cv::Mat maskf((int)outputShapeDecoder[0][2], (int)outputShapeDecoder[0][3], CV_32F,
                    outputDecoder[0].data() + k * (int)outputShapeDecoder[0][2] * (int)outputShapeDecoder[0][3] + b * maskSize);
      cv::Mat maskResized;
      cv::resize(maskf, maskResized, imageSize, 0, 0, cv::INTER_LINEAR);
      cv::Mat mask(imageSize.height, imageSize.width, CV_8UC1, cv::Scalar(0));
      mask.setTo(255, maskResized > 0);
      masks.push_back(mask);
    }
  }
  preprocessingEnd();
  end = std::chrono::steady_clock::now();
  std::cout << "changeThreshold sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  return std::make_tuple(masks, boxes);
}


