template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::NeuralNetModel(float delta_t, std::array<float2, C_DIM> control_rngs) {
  dt_ = delta_t;
  if(control_rngs.size() == C_DIM) {
    for(int i = 0; i < C_DIM; i++) {
      control_rngs_[i] = control_rngs[i];
    }
  } else {
    std::cerr << "ERROR: wrong size control rngs" << std::endl;
    exit(1);
  }
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::NeuralNetModel(float delta_t) {
  for(int i = 0; i < C_DIM; i++) {
    control_rngs_[i].x = -FLT_MAX;
    control_rngs_[i].y = FLT_MAX;
  }
}

template<int S_DIM, int C_DIM, int K_DIM, int... layer_args>
NeuralNetModel<S_DIM, C_DIM, K_DIM, layer_args...>::~NeuralNetModel() {

}
