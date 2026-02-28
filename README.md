## Segment Anything Model 3 CPP Wrapper for macOS

This code is to run [Segment Anything Model 3](https://github.com/facebookresearch/sam3) ONNX models in c++ code and implemented on the macOS app [RectLabel](https://rectlabel.com).

<video src="https://github.com/user-attachments/assets/812776c3-bfad-4f80-99e1-6141b21c024b" controls="controls" muted="muted" class="width-fit" style="max-height:640px; min-height: 200px"></video>

Install SAM 3.

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

Add Apple CPU support from [this PR](https://github.com/facebookresearch/sam3/pull/258).
```bash
gh pr checkout 258, ok
```

Download SAM 3 model from Hugging Face [repo](https://huggingface.co/facebook/sam3) and put them into sam3-model folder. 

Install Segment Anything Model 3 CPP Wrapper.
```bash
git clone https://github.com/ryouchinsa/sam3-cpp-macos.git
cd sam3-cpp-macos
```

Export ONNX models.
```bash
python export.py --all --model-path ../sam3-model
```

Download exported [SAM 3 ONNX models](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam3.zip). 

Build and run.

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/Users/ryo/Downloads/onnxruntime-osx-universal2-1.23.2
cmake --build build

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder_q.onnx" -text_encoder="sam3/text-encoder_q.onnx" -geometry_encoder="sam3/geometry-encoder_q.onnx" -decoder="sam3/decoder_q.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu" -text="zebra" -threshold=0.5

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder_q.onnx" -text_encoder="sam3/text-encoder_q.onnx" -geometry_encoder="sam3/geometry-encoder_q.onnx" -decoder="sam3/decoder_q.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu" -boxes="pos:124,113,183,329" -threshold=0.5
```
