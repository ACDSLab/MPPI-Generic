include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# Install all library header files
install(
        DIRECTORY include/MPPI-Generic
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
)

add_subdirectory(src)
add_subdirectory(examples)
