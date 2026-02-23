import onnx
from onnx import helper, TensorProto

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])

b = helper.make_tensor(
    name="b",
    data_type=TensorProto.FLOAT,
    dims=[1, 3],
    vals=[1.0, 2.0, 3.0],
)

add = helper.make_node("Add", inputs=["x", "b"], outputs=["t"], name="Add1")
relu = helper.make_node("Relu", inputs=["t"], outputs=["y"], name="Relu1")

graph = helper.make_graph(
    nodes=[add, relu],
    name="TinyGraph",
    inputs=[x],
    outputs=[y],
    initializer=[b],
)

model = helper.make_model(graph, producer_name="onnx_example")
onnx.checker.check_model(model)
onnx.save(model, "example.onnx")
print("Wrote example.onnx")
