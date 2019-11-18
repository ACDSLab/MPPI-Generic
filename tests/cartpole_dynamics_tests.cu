//
// Created by mgandhi3 on 10/4/19.
//

#include <gtest/gtest.h>
#include "cartpole.cuh"


TEST(Dummy, One) {

    auto CP = Cartpole(0.1, 1, 1, 1);
    std::cout << "Hello Testing!" << std::endl;
    FAIL();
}