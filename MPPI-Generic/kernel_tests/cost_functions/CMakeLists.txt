set(LIBRARY_NAME ar_standard_cost)
SET (LIB_MAJOR 0)
SET (LIB_MINOR 0)
SET (LIB_RELEASE 1)

add_library(${LIBRARY_NAME} SHARED
        ar_standard_cost.cu)

target_link_libraries( ${LIBRARY_NAME}
        ${YAML_CPP_LIBRARIES}
        )

if(NOT ${CURAND_LIBRARY} MATCHES "NOTFOUND")
    target_link_libraries( ${LIBRARY_NAME}
            ${CURAND_LIBRARY}
            )
endif()

SET (_soversion ${LIB_MAJOR}.${LIB_MINOR}.${LIB_RELEASE})

install(TARGETS ${LIBRARY_NAME}
        # IMPORTANT: Add the library to the "export-set"
        EXPORT ${PROJECT_NAME}-targets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib/${CMAKE_PROJECT_NAME}
        )