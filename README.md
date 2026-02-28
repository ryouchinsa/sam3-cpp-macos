## Segment Anything Model 3 CPP Wrapper for macOS

This code is to run [Segment Anything Model 3](https://github.com/facebookresearch/sam3) ONNX models in c++ code and implemented on the macOS app [RectLabel](https://rectlabel.com).

<video src="https://github.com/user-attachments/assets/459f0f9e-e74a-41f7-912a-dcf5b801625e" controls="controls" muted="muted" class="width-fit" style="max-height:640px; min-height: 200px"></video>

Install SAM 3 and add Apple CPU support from [this PR](https://github.com/facebookresearch/sam3/pull/258).

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
gh pr checkout 258
pip install -e .
```

Install Segment Anything Model 3 CPP Wrapper.
```bash
git clone https://github.com/ryouchinsa/sam3-cpp-macos.git
cd sam3-cpp-macos
```

Download SAM 3 model from [repo](https://huggingface.co/facebook/sam3) and put them into sam3-model folder. 

Export ONNX models. This script is originated from [sam3-image](https://github.com/jamjamjon/usls/tree/main/scripts/sam3-image).
```bash
python export.py --all --model-path sam3-model
```

Download exported SAM 3 ONNX models from [repo](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam3.zip). 

Download ONNX Runtime from [repo](https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-universal2-1.23.2.tgz).

Download tokenizers-cpp from [repo](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/tokenizers-cpp.zip).

Build and run.

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/Users/ryo/Downloads/onnxruntime-osx-universal2-1.23.2 -DTOKENIZERS_ROOT_DIR=/Users/ryo/Downloads/tokenizers-cpp

cmake --build build

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3-model/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu" -text="zebra" -threshold=0.5

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3-model/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu" -boxes="pos:124,113,183,329" -threshold=0.5
```
