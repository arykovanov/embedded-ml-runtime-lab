"""Generate the single-operator ONNX model used to test ryk::RykRelu."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main():
    node = helper.make_node(
        "RykRelu",
        inputs=["images"],
        outputs=["logits"],
        name="relu_test",
        domain="ryk",
    )
    graph = helper.make_graph(
        nodes=[node],
        name="RykReluTest",
        inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT, [1,28,28])],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1,28,28])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("ryk", 1)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    output_path = Path(__file__).resolve().with_name("relu_test.onnx")
    onnx.save(model, output_path)
    print(f"Saved test model to {output_path}")


if __name__ == "__main__":
    main()
