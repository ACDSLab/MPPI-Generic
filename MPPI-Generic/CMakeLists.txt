include_directories(include)

# Install all library header files
install(
        DIRECTORY include/MPPI-Generic
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
)

add_subdirectory(src)
add_subdirectory(examples)
#add_subdirectory(test)
