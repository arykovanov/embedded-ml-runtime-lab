idf_component_register(
    SRCS 
        "edgeml.cpp"
        "../../cpp/tensor_utils.cpp"

    INCLUDE_DIRS
        "."
        "../../cpp/"

    EMBED_FILES 
        "../../model/lenet_quantized.tflite"
        "../../model/sample_images.bin"
)
