add_executable(ddp_tests ../test_main.cpp ddp_test.cu)
target_link_libraries(ddp_tests gtest
        gmock
        gtest_main
        cartpole_mppi
        cnpy)
gtest_discover_tests(ddp_tests)
set_target_properties(ddp_tests PROPERTIES FOLDER test)
