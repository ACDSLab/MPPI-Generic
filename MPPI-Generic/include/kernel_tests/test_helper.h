//
// Created by mgandhi3 on 1/13/20.
//

#ifndef MPPIGENERIC_KERNEL_TESTS_TEST_HELPER_H
#define MPPIGENERIC_KERNEL_TESTS_TEST_HELPER_H
#include <gtest/gtest.h>

inline void array_expect_float_eq(const std::vector<float>& known,const std::vector<float>& compute, int size) {
    EXPECT_EQ(compute.size(), size) << "The computed vector size is not the given size!";
    EXPECT_EQ(known.size(), compute.size()) << "Two vectors are not the same size!";
    for (int i = 0; i < size; i++) {
        EXPECT_FLOAT_EQ(known[i], compute[i]) << "Failed at index: " << i;
    }
}

inline void array_expect_float_eq(const float known, const std::vector<float>& compute, const int size) {
    EXPECT_EQ(compute.size(), size) << "The computed vector size is not the given size!";
    for (int i = 0; i < size; i++) {
        EXPECT_FLOAT_EQ(known, compute[i]) << "Failed at index: " << i;
    }
}



#endif //MPPIGENERIC_KERNEL_TESTS_TEST_HELPER_H
