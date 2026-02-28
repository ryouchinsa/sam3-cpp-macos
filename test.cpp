#include <gflags/gflags.h>
#include <thread>
#include <opencv2/opencv.hpp>
#include "sam3.h"

DEFINE_string(vision_encoder, "sam3/vision-encoder.onnx", "Path to the viion encoder model");
DEFINE_string(text_encoder, "sam3/text-encoder.onnx", "Path to the text encoder model");
DEFINE_string(geometry_encoder, "sam3/geometry-encoder.onnx", "Path to the geometry encoder model");
DEFINE_string(decoder, "sam3/decoder.onnx", "Path to the decoder model");
DEFINE_string(tokenizer, "sam3/tokenizer.json", "Path to the tokenizer");
DEFINE_string(text, "", "Text prompt");
DEFINE_string(boxes, "", "Boxes prompt");
DEFINE_double(threshold, 0.5, "Threshold for detections");
DEFINE_string(image, "david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg", "Path to the image");
DEFINE_string(device, "cpu", "cpu or cuda:0(1,2,3...)");

int main(int argc, char** argv) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  Sam3 sam3;
  std::cout<<"loadModel started"<<std::endl;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  bool successLoadModel = sam3.loadModel(FLAGS_vision_encoder, FLAGS_text_encoder, FLAGS_geometry_encoder, FLAGS_decoder, FLAGS_tokenizer, std::thread::hardware_concurrency(), FLAGS_device);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  if(!successLoadModel){
    std::cout<<"loadModel error"<<std::endl;
    return 1;
  }
  std::cout<<"preprocessImage started"<<std::endl;
  cv::Mat image = cv::imread(FLAGS_image, cv::IMREAD_COLOR);
  cv::Size imageSize = cv::Size(image.cols, image.rows);
  cv::Size inputSize = sam3.getInputSize();
  cv::resize(image, image, inputSize);
  begin = std::chrono::steady_clock::now();
  bool successPreprocessImage = sam3.preprocessImage(image);
  end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  if(!successPreprocessImage){
    std::cout<<"preprocessImage error"<<std::endl;
    return 1;
  }
  std::cout<<"Encode text started"<<std::endl;
  begin = std::chrono::steady_clock::now();
  bool successEncodeText = sam3.encodeText(FLAGS_text);
  end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  if(!successEncodeText){
    std::cout<<"Encode text error"<<std::endl;
    return 1;
  }
  std::cout<<"Encode boxes started"<<std::endl;
  begin = std::chrono::steady_clock::now();
  auto [rects, labels] = parse_box_prompts(FLAGS_boxes);
  normalizeRects(&rects, imageSize);
  bool successEncodeBoxes = sam3.encodeBoxes(rects, labels);
  end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  if(!successEncodeBoxes){
    std::cout<<"Encode boxes error"<<std::endl;
    return 1;
  }
  std::cout<<"Decode started"<<std::endl;
  begin = std::chrono::steady_clock::now();
  float threshold = FLAGS_threshold;
  bool skipDecode = false;
  auto [masks, boxes] = sam3.decode(threshold, imageSize, skipDecode);
  end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  if(masks.size() == 0){
    std::cout<<"Decode error"<<std::endl;
    return 1;
  }
  std::cout<<"Found "<<masks.size()<<std::endl;
  for(int i = 0; i < masks.size(); i++){
    std::string fileName = "mask" + std::to_string(i) + ".png";
    cv::imwrite(fileName, masks[i]);
  }
  return 0;
}
