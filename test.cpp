#include <gflags/gflags.h>
#include <thread>
#include <opencv2/opencv.hpp>
#include "sam3.h"

DEFINE_string(vision_encoder, "sam3/vision-encoder.onnx", "Path to the viion encoder model");
DEFINE_string(text_encoder, "sam3/text-encoder.onnx", "Path to the text encoder model");
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
  std::chrono::steady_clock::time_point begin, end, begin_total, end_total; 
  std::cout<<"loadModel started"<<std::endl;
  begin = std::chrono::steady_clock::now();
  bool successLoadModel = sam3.loadModel(FLAGS_vision_encoder, FLAGS_text_encoder, FLAGS_decoder, FLAGS_tokenizer, std::thread::hardware_concurrency(), FLAGS_device);
  end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  if(!successLoadModel){
    std::cout<<"loadModel error"<<std::endl;
    return 1;
  }
  begin_total = std::chrono::steady_clock::now();
  std::cout<<"preprocessImage started"<<std::endl;
  begin = std::chrono::steady_clock::now();
  cv::Mat image = cv::imread(FLAGS_image, cv::IMREAD_COLOR);
  cv::Size imageSize = cv::Size(image.cols, image.rows);
  cv::Size inputSize = sam3.getInputSize();
  cv::resize(image, image, inputSize);
  end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
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
  std::vector<std::string> text_list = split(FLAGS_text, ',');
  auto [rects_list, labels_list] = parse_box_list_prompts(FLAGS_boxes, imageSize);
  sam3.alignTextsAndBoxes(&text_list, &rects_list, &labels_list);
  bool successEncodeText = sam3.encodeText(text_list);
  end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  if(!successEncodeText){
    std::cout<<"Encode text error"<<std::endl;
    return 1;
  }
  std::cout<<"Decode started"<<std::endl;
  float threshold = FLAGS_threshold;
  bool skipDecode = false;
  auto [masks, boxes] = sam3.decode(rects_list, labels_list, threshold, imageSize, skipDecode);
  if(masks.size() == 0){
    std::cout<<"Decode error"<<std::endl;
    return 1;
  }
  std::cout<<"Found "<<masks.size()<<std::endl;
  end_total = std::chrono::steady_clock::now();
  std::cout << "predict sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end_total - begin_total).count()) / 1000000.0 <<std::endl;
  for(int i = 0; i < masks.size(); i++){
    std::string fileName = "mask" + std::to_string(i) + ".png";
    cv::imwrite(fileName, masks[i]);
  }
  return 0;
}
