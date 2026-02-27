# import onnx
# from onnx import helper, TensorProto

# x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
# y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])

# b = helper.make_tensor(
#     name="b",
#     data_type=TensorProto.FLOAT,
#     dims=[1, 3],
#     vals=[1.0, 2.0, 3.0],
# )

# add = helper.make_node("Add", inputs=["x", "b"], outputs=["t"], name="Add1")
# relu = helper.make_node("Relu", inputs=["t"], outputs=["y"], name="Relu1")

# graph = helper.make_graph(
#     nodes=[add, relu],
#     name="TinyGraph",
#     inputs=[x],
#     outputs=[y],
#     initializer=[b],
# )

# model = helper.make_model(graph, producer_name="onnx_example")
# onnx.checker.check_model(model)
# onnx.save(model, "example.onnx")
# print("Wrote example.onnx")

import onnx
from onnx import helper, TensorProto

# x: [1, 3]
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])

# Константы (initializers)
b1 = helper.make_tensor(
    name="b1",
    data_type=TensorProto.FLOAT,
    dims=[1, 3],
    vals=[1.0, 2.0, 3.0],
)

b2 = helper.make_tensor(
    name="b2",
    data_type=TensorProto.FLOAT,
    dims=[1, 3],
    vals=[0.5, 1.5, -2.0],
)

# Branch A: a = Relu(x + b1)
add1 = helper.make_node("Add", inputs=["x", "b1"], outputs=["t1"], name="Add_branch")
relu = helper.make_node("Relu", inputs=["t1"], outputs=["a"], name="Relu_branch")

# Branch B: c = Sigmoid(x * b2)
mul = helper.make_node("Mul", inputs=["x", "b2"], outputs=["t2"], name="Mul_branch")
sigm = helper.make_node("Sigmoid", inputs=["t2"], outputs=["c"], name="Sigmoid_branch")

# Merge: y = a + c
add2 = helper.make_node("Add", inputs=["a", "c"], outputs=["y"], name="Add_merge")

graph = helper.make_graph(
    nodes=[add1, relu, mul, sigm, add2],
    name="BranchyGraph",
    inputs=[x],
    outputs=[y],
    initializer=[b1, b2],
)

model = helper.make_model(graph, producer_name="onnx_branch_example")
onnx.checker.check_model(model)
onnx.save(model, "branchy.onnx")
print("Wrote branchy.onnx")
