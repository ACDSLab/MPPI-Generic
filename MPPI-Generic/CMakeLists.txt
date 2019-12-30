include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/MPPI-Generic/include/controllers
        ${CMAKE_SOURCE_DIR}/MPPI-Generic/include/cost_functions
        ${CMAKE_SOURCE_DIR}/MPPI-Generic/include/dynamics
        ${CMAKE_SOURCE_DIR}/MPPI-Generic/include/mppi_core
        ${CMAKE_SOURCE_DIR}/MPPI-Generic/include/utils
        )

# Install all library header files
install(
        DIRECTORY include/MPPI-Generic
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
)

add_subdirectory(src)
add_subdirectory(examples)
#add_subdirectory(test)
