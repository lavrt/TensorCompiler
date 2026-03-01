import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

# Shapes
# x: NCHW = [1, 1, 5, 5]
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 5, 5])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])  # конечный выход

# ---- Initializers ----
# Conv weights: [out_channels=2, in_channels=1, kH=3, kW=3]
Wc = numpy_helper.from_array(
    np.arange(2 * 1 * 3 * 3, dtype=np.float32).reshape(2, 1, 3, 3),
    name="Wc"
)
# Conv bias: [2]
bc = numpy_helper.from_array(np.array([0.1, -0.2], dtype=np.float32), name="bc")

# Add constant for after conv: shape [1,2,3,3] (conv output will be 3x3 with pads=0,stride=1,k=3)
add_const = numpy_helper.from_array(
    np.ones((1, 2, 3, 3), dtype=np.float32),
    name="add_const"
)

# Mul constant: same shape
mul_const = numpy_helper.from_array(
    (2.0 * np.ones((1, 2, 3, 3), dtype=np.float32)),
    name="mul_const"
)

# Flattened size after conv: 1*2*3*3 = 18
# MatMul: A [1,18] * B [18,3] -> [1,3]
Wmm = numpy_helper.from_array(
    np.arange(18 * 3, dtype=np.float32).reshape(18, 3) / 10.0,
    name="Wmm"
)

# Gemm: A [1,18], B [3,18] with transB=1 -> uses B^T [18,3], C [1,3]
Bg = numpy_helper.from_array(
    np.arange(3 * 18, dtype=np.float32).reshape(3, 18) / 20.0,
    name="Bg"
)
Cg = numpy_helper.from_array(
    np.array([[0.5, 0.0, -0.5]], dtype=np.float32),
    name="Cg"
)

# ---- Nodes ----

# Conv with attributes
conv = helper.make_node(
    "Conv",
    inputs=["x", "Wc", "bc"],
    outputs=["conv_out"],
    name="Conv1",
    # attributes:
    kernel_shape=[3, 3],
    strides=[1, 1],
    dilations=[1, 1],
    pads=[0, 0, 0, 0],
    group=1,
)

relu = helper.make_node(
    "Relu",
    inputs=["conv_out"],
    outputs=["relu_out"],
    name="Relu1"
)

add = helper.make_node(
    "Add",
    inputs=["relu_out", "add_const"],
    outputs=["add_out"],
    name="Add1"
)

mul = helper.make_node(
    "Mul",
    inputs=["add_out", "mul_const"],
    outputs=["mul_out"],
    name="Mul1"
)

# Flatten to [1,18]
flatten = helper.make_node(
    "Flatten",
    inputs=["mul_out"],
    outputs=["flat"],
    name="Flatten1",
    axis=1
)

# Branch A: MatMul
matmul = helper.make_node(
    "MatMul",
    inputs=["flat", "Wmm"],
    outputs=["mm_out"],
    name="MatMul1"
)

# Branch B: Gemm with attributes
gemm = helper.make_node(
    "Gemm",
    inputs=["flat", "Bg", "Cg"],
    outputs=["gemm_out"],
    name="Gemm1",
    alpha=1.0,
    beta=1.0,
    transA=0,
    transB=1
)

# Merge: Add the two branches => y
merge = helper.make_node(
    "Add",
    inputs=["mm_out", "gemm_out"],
    outputs=["y"],
    name="Add_merge"
)

graph = helper.make_graph(
    nodes=[conv, relu, add, mul, flatten, matmul, gemm, merge],
    name="AllOpsWithAttrs",
    inputs=[x],
    outputs=[y],
    initializer=[Wc, bc, add_const, mul_const, Wmm, Bg, Cg],
)

model = helper.make_model(graph, producer_name="onnx_all_ops_example")
onnx.checker.check_model(model)
onnx.save(model, "all_ops_attrs.onnx")
print("Wrote all_ops_attrs.onnx")
