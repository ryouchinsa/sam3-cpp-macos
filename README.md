## Segment Anything Model 3 CPP Wrapper for macOS and Ubuntu GPU

This code is to run [Segment Anything Model 3](https://github.com/facebookresearch/sam3) ONNX models in c++ code and implemented on the macOS app [RectLabel](https://rectlabel.com).

We recommend working through this blog post side-by-side with the [Google Colab notebook](https://colab.research.google.com/github/ryouchinsa/sam3-cpp-macos/blob/master/notebooks/sam3_cpp_macos.ipynb
).

<video src="https://github.com/user-attachments/assets/459f0f9e-e74a-41f7-912a-dcf5b801625e" controls="controls" muted="muted" class="width-fit" style="max-height:640px; min-height: 200px"></video>

Install [CUDA, cuDNN, PyTorch, and ONNX Runtime](https://rectlabel.com/pytorch/).

Install Segment Anything Model 3 CPP Wrapper.

```bash
git clone https://github.com/ryouchinsa/sam3-cpp-macos.git
```

Install SAM 3.

For macOS, add Apple CPU support from [this PR](https://github.com/facebookresearch/sam3/pull/258).
```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
gh pr checkout 258
pip install -e .
cd ..
```

For Ubuntu GPU, use numpy>=2.0.

```bash
git clone https://github.com/facebookresearch/sam3.git
cp  sam3-cpp-macos/pyproject.toml sam3/
cd sam3
pip install -e .
cd ..
```
Download SAM 3 model from [Hugging Face](https://huggingface.co/facebook/sam3).

```bash
hf auth login
hf download facebook/sam3 model.safetensors tokenizer.json
```

Export ONNX models. This script is originated from [sam3-image](https://github.com/jamjamjon/usls/tree/main/scripts/sam3-image). Edit --model-path according to your downloaded huggingface path.

```bash
cd sam3-cpp-macos
python export.py --all --model-path sam3-model
cd ..
```

If you skip exporting, download exported SAM 3 ONNX models from [Hugging Face](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam3.zip). 

Install tokenizers-cpp.

For macOS, download tokenizers-cpp from [Hugging Face](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/tokenizers-cpp.zip).

For Ubuntu GPU.

```bash
git clone --recursive https://github.com/mlc-ai/tokenizers-cpp.git

cp /content/sam3-cpp-macos/tokenizers-cpp/CMakeLists.txt .
cp /content/sam3-cpp-macos/tokenizers-cpp/msgpack/CMakeLists.txt msgpack
cp /content/sam3-cpp-macos/tokenizers-cpp/sentencepiece/CMakeLists.txt sentencepiece

apt update
apt install -y curl gcc make
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

import os
os.environ['CARGO_HOME'] = '/root/.cargo'
os.environ['PATH'] = f"{os.environ['CARGO_HOME']}/bin:{os.environ['PATH']}"

rustc --version
cargo --version

cd tokenizers-cpp/example
./build_and_run.sh
cd ..

mkdir lib
cp ./example/build/tokenizers/sentencepiece/src/libsentencepiece.a lib/
cp ./example/build/tokenizers/libtokenizers_c.a lib/
cp ./example/build/tokenizers/libtokenizers_cpp.a lib/
cd ..
```

Build and run.

```bash
cd sam3-cpp-macos
# macOS
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/Users/ryo/Downloads/onnxruntime-osx-universal2-1.23.2 -DTOKENIZERS_ROOT_DIR=/Users/ryo/Downloads/tokenizers-cpp
# Ubuntu GPU
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/content/onnxruntime-linux-x64-gpu-1.23.2 -DTOKENIZERS_ROOT_DIR=/content/tokenizers-cpp

cmake --build build

# macOS
./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu" -text="zebra" -threshold=0.5

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu" -boxes="pos:124,113,183,329" -threshold=0.5

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu" -text="zebra" -boxes="pos:379,454,329,297" -threshold=0.5

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu" -text="zebra,water" -threshold=0.5

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu" -boxes="pos:0,0,364,187-pos:379,454,329,297" -threshold=0.5

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu" -text="tree,zebra" -boxes="pos:0,0,364,187-pos:379,454,329,297" -threshold=0.5

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu" -text="zebra,water,tree" -threshold=0.25

# Ubuntu GPU
./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cuda:0" -text="zebra" -threshold=0.5

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cuda:0" -boxes="pos:124,113,183,329" -threshold=0.5

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cuda:0" -text="zebra" -boxes="pos:379,454,329,297" -threshold=0.5

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cuda:0" -text="zebra,water" -threshold=0.5

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cuda:0" -boxes="pos:0,0,364,187-pos:379,454,329,297" -threshold=0.5

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cuda:0" -text="tree,zebra" -boxes="pos:0,0,364,187-pos:379,454,329,297" -threshold=0.5

./build/sam3_cpp_test -vision_encoder="sam3/vision-encoder.onnx" -text_encoder="sam3/text-encoder.onnx" -geometry_encoder="sam3/geometry-encoder.onnx" -decoder="sam3/decoder.onnx" -tokenizer="sam3/tokenizer.json" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cuda:0" -text="zebra,water,tree" -threshold=0.25
```
