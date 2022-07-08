# Arrow integration

add_library(
        arrow
        OBJECT
        file_lance.cc
        reader.cc
        type.cc
        type.h
        writer.cc
)
target_include_directories(arrow SYSTEM PRIVATE ${Protobuf_INCLUDE_DIR})

add_lance_test(api_test)
add_lance_test(arrow_dataset_test)
add_lance_test(writer_test)
add_lance_test(type_test)
