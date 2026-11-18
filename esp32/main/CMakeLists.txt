idf_component_register(
    SRCS 
        "edgeml.cpp"
        "load_model.cpp"
        "../../cpp/tensor_utils.cpp"

    INCLUDE_DIRS
        "."
        "../../cpp/"

    EMBED_FILES 
        "../../model/sample_images.bin"

    REQUIRES
        nvs_flash
        esp_timer
)
