add_library(
        lance_io_exec
        OBJECT
        base.h
        filter.cc
        filter.h
        limit.cc
        limit.h
        project.cc
        project.h
        scan.cc
        scan.h
        take.cc
        take.h
        base.cc)
target_include_directories(lance_io_exec SYSTEM PRIVATE ${Protobuf_INCLUDE_DIR})
add_dependencies(lance_io_exec format)

add_lance_test(filter_test)
add_lance_test(project_test)
add_lance_test(limit_test)
add_lance_test(scan_test)
