# Internal I/O

add_library(
        io
        OBJECT
        endian.h
        pb.cc
        pb.h
        reader.cc
        reader.h
        scanner.cc
        scanner.h
        writer.cc
        writer.h
)
target_include_directories(io SYSTEM PRIVATE ${Protobuf_INCLUDE_DIR})
# Depend on lance::format to generate protobuf
add_dependencies(io format)
