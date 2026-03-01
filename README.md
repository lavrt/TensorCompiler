# TensorCompiler

A small educational tensor compiler project.  
Current stage: read an ONNX model, build an internal IR (computational graph), and optionally dump it to GraphViz `.dot`.

## Features (current)
- Parse `.onnx` files (protobuf + ONNX)
- Convert ONNX graph to internal IR (`ir::Graph`, `ir::Node`, `ir::Value`)
- Track:
  - graph inputs / outputs
  - initializers (constants)
  - producer/consumer links
  - node attributes (`int`, `float`, `string`, `ints`, `floats`)
- GraphViz dump to `.dot` (visualization)


Supported ops (at least):
- `Add`, `Mul`, `Conv`, `Relu`, `MatMul`, `Gemm`  
(Other ops may appear and can be handled as `Unknown`.)

## Requirements
- Linux
- CMake >= 3.11
- Conan 2
- GCC 14 (`/usr/bin/g++-14`) for building this project

## Get the code

```bash
git clone https://github.com/lavrt/TensorCompiler
cd TensorCompiler
```

## Build
From the project root:
```bash
conan install . -of build -s build_type=Release --build=missing
cmake --preset conan-release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14
cmake --build --preset conan-release -j
```

## Run
```bash
./build/run --help
./build/run --onnx model.onnx
```

## GraphViz dump

```bash
./build/run --onnx model.onnx --dump graph.dot
dot -Tpng graph.dot -o graph.png
```
## Tests

```bash
ctest --preset conan-release --output-on-failure
```