idf_component_register(
    SRCS 
    "ryk_http_server.cpp"
        "ryk_main.cpp"
        "ryk_model.cpp"
        "ryk_model_data.cpp"
        "ryk_wifi.cpp"
        "../../cpp/tensor_utils.cpp"

    INCLUDE_DIRS
        "."
        "../../cpp/"

    EMBED_FILES 
        "../../model/sample_images.bin"

     PRIV_REQUIRES
        esp_http_server
        esp_timer
        esp_wifi
        nvs_flash
        wifi_provisioning
        bt
)
