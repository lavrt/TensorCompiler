# TensorCompiler
```bash
conan install . -of build -s build_type=Release --build=missing
cmake --preset conan-release
cmake --build --preset conan-release -j
```

```bash
cd ./onnx_maker/
source .venv/bin/activate
python3 make_onnx.py
netron example.onnx
deactivate
```