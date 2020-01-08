#pragma once

#ifndef AR_STANDARD_COST_CUH_
#define AR_STANDARD_COST_CUH_

#include <cost_functions/cost.cuh>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <cuda_runtime.h>

class ARStandardCost : public Cost {
public:

  typedef struct {
    float desired_speed;
    float speed_coeff;
    float track_coeff;
    float max_slip_ang;
    float slip_penalty;
    float track_slop;
    float crash_coeff;
    float steering_coeff;
    float throttle_coeff;
    float boundary_threshold;
    float discount;
    int num_timesteps;
    int grid_res;
    float3 r_c1;
    float3 r_c2;
    float3 trs;
  } ARStandardCostParams;

  /**
   * Constructor
   * @param width
   * @param height
   */
  ARStandardCost(int width, int height, cudaStream_t stream=0);

  /**
   *
   */
  ~ARStandardCost();

  /**
   * allocates all the extra cuda memory
   */
  void GPUSetup();

  /**
   * Deallocates the allocated cuda memory for an object
   * TODO make a generic version of this
   */
  void freeCudaMem();

  void setParams(ARStandardCostParams params);
  ARStandardCostParams getParams();

  /**
   * Copies the parameters to the GPU object
   */
  void paramsToDevice();

  /**
   * @brief Allocates memory to cuda array which is bound to a texture.
   *
   * Allocates an array using the special cudaMallocArray function.
   * The size of the array allocated by this function is determined based on
   * the width and height of the costmap. This function is called by the constructor
   */
  //void allocateTexMem();

  /**
  * @brief Initializes the host side costmap to all zeros.
  *
  * Initializes a float4 vector to the correct width and height and sets every value to zero.
  * This function is called by both constructors.
  */
  //void initCostmap();

  /**
   * Converts the passed in costmap to a CUDA texture
   * @param costmap to be converted
   * @param channel
   */
  //void costmapToTexture(float* costmap, int channel = 0);

  /**
  * @brief Binds the member variable costmap to a CUDA texture.
  */
  //void costmapToTexture();

  /**
   * @brief Updates the current costmap coordinate transform.
   * @param h Matrix representing a transform from world to (offset) costmap coordinates.
   * @param trs Array representing the offset.
   */
  //void updateTransform(Eigen::MatrixXf h, Eigen::ArrayXf trs);

  /**
   * @brief Loads track data from a file.
   * @param C-string representing the path to the costmap data file.
   * @param h Matrix representing a transform from world to (offset) costmap coordinates.
   * @param trs Array representing the offset.
   */
  //std::vector<float4> loadTrackData(std::string map_path, Eigen::Matrix3f &R, Eigen::Array3f &trs);

  /**
   * @brief Return what the desired speed is set to.
   */
  //float getDesiredSpeed();

  /**
   * @brief Sets the desired speed of the vehicle.
   * @param desired_speed The desired speed.
   */
  //void setDesiredSpeed(float desired_speed);

  /**
   *@brief Initializes the debug window for a default 20x20 meter window.
   */
  //void debugDisplayInit();

  /**
   * @brief Initialize and allocate memory for debug window display
   */
  //void debugDisplayInit(int width_m, int height_m, int ppm);

  /**
   * @brief Display the debug view centered around x and y.
   * @param x float representing the current x-coordinate
   * @param y float representing the current y-coordinate
   */
  //cv::Mat getDebugDisplay(float x, float y, float heading);

  /**
   *
   * @param description
   * @param data
   */
  //void updateCostmap(std::vector<int> description, std::vector<float> data);

  /**
   *
   * @param description
   * @param data
   */
  //void updateObstacles(std::vector<int> description, std::vector<float> data);

  /**
   * @brief Returns whether or not the vehicle has crashed or not
   */
  //__host__ __device__ void getCrash(float* state, int* crash);

  /**
   * @brief Compute the control cost
   */
  //__host__ __device__ float getControlCost(float* u, float* du, float* vars);

  /**
   * @brief Compute the cost for achieving a desired speed
   */
  //__host__ __device__ float getSpeedCost(float* s, int* crash);

  /**
   * @brief Compute a penalty term for crashing
   */
  //__host__ __device__ float getCrashCost(float* s, int* crash, int num_timestep);

  /**
   * @brief Compute some cost terms that help stabilize the car.
   */
  //__host__ __device__ float getStabilizingCost(float* s);

  /**
   * @brief Compute a coordinate transform going from world to costmap coordinates.
   */
  //__host__ __device__ void coorTransform(float x, float y, float* u, float* v, float* w);

  /**
   * @brief Compute the current track cost based on the costmap.
   * Requires using CUDA texture memory so can only be run on the GPU
   */
  //__device__ float getTrackCost(float* s, int* crash);

  /**
   * @brief Compute all of the individual cost terms and adds them together.
   */
  //__host__ __device__ float computeCost(float* s, float* u, float* du, float* vars, int* crash, int t);

  /**
   * @brief Computes the terminal cost from a state
   */
  //__host__ __device__ float terminalCost(float* s);

  ARStandardCost* cost_device_;

protected:

  //Constant variables
  const float FRONT_D = 0.5; ///< Distance from GPS receiver to front of car.
  const float BACK_D = -0.5; ///< Distance from GPS receiver to back of car.

  bool l1_cost_; //Whether to use L1 speed cost (if false it is L2)

  //Primary variables
  int width_, height_; ///< Width and height of costmap.
  ARStandardCostParams params_; ///< object copy of params
  cudaArray *costmapArray_d_; ///< Cuda array for texture binding.
  cudaChannelFormatDesc channelDesc_; ///< Cuda texture channel description.
  cudaTextureObject_t costmap_tex_; ///< Cuda texture object.
  std::vector<float4> track_costs_;

  ARStandardCost* cost_d_;

  //Debugging variables
  float* debug_data_; ///< Host array for holding debug info.
  //float* debug_data_d_; ///< Device array for holding debug info.
  int debug_img_width_; ///Width (in meters) of area imaged by debug view.
  int debug_img_height_; ///< Height (in meters) of area imaged by debug view.
  int debug_img_ppm_; ///< Pixels per meter for resolution of debug view.
  int debug_img_size_; ///< Number of pixels in the debug image.
  bool debugging_; ///< Indicator for if we're in debugging mode

protected:

};

#endif // AR_STANDARD_COST_CUH_
