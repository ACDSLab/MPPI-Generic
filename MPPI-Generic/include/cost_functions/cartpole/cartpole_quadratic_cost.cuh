#pragma once

#ifndef CARTPOLE_QUADRATIC_COST_CUH_
#define CARTPOLE_QUADRATIC_COST_CUH_

#include <cost_functions/cost.cuh>
#include <utils/file_utils.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <cuda_runtime.h>

class CartpoleQuadraticCost : public Cost {
public:

    typedef struct {
        float cart_position_coeff = 1;
        float cart_velocity_coeff = 1;
        float pole_angle_coeff = 10;
        float pole_angular_velocity_coeff = 1;
        float control_force_coeff = 1;
        float terminal_cost_coeff = 100;
    } Params;

    /**
     * Constructor
     * @param width
     * @param height
     */
    CartpoleQuadraticCost(cudaStream_t stream=0);

    /**
     *
     */
    ~CartpoleQuadraticCost();

    /**
     * allocates all the extra cuda memory
     */
    void GPUSetup();

    /**
     * Deallocates the allocated cuda memory for an object
     */
    void freeCudaMem();

    /**
     * Updates GPU if allocated
     * @param params
     */
    inline __host__ __device__ void setParams(Params params);

    /**
     *
     * @return current parameters for the system
     */
    inline __host__ __device__ Params getParams() {return params_;}


    /**
     * Copies the parameters to the GPU object
     */
    void paramsToDevice();

    /**
     * @brief Compute the control cost
     */
    __host__ __device__ float getControlCost(float* u, float* du, float* vars);

    /**
     * @brief Compute the state cost
     */
    __host__ __device__ float getStateCost(float* s);


    /**
     * @brief Compute all of the individual cost terms and adds them together.
     */
    __host__ __device__ float computeRunningCost(float* s, float* u, float* du, float* vars);

    /**
     * @brief Compute the terminal cost of the system
     */
     __host__ __device__ float computeTerminalCost(float *s);

    CartpoleQuadraticCost* cost_d_ = nullptr;

protected:

    Params params_; ///< object copy of params

};



#if __CUDACC__
#include "cartpole_quadratic_cost.cu"
#endif

#endif // CARTPOLE_QUADRATIC_COST_CUH_// Include the cart pole cost.