#include "lstm_lstm_helper.cuh"

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>(cudaStream_t stream)
  : Managed(stream)
{
  init_model_ = std::make_shared<INIT_T>(stream);
  lstm_ = new LSTM_T(stream);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>(std::string path,
                                                                                         cudaStream_t stream)
  : Managed(stream)
{
  init_model_ = std::make_shared<INIT_T>(stream);
  lstm_ = new LSTM_T(stream);
}

template <class INIT_T, class LSTM_T, int INITIAL_LEN>
void LSTMLSTMHelper<INIT_T, LSTM_T, INITIAL_LEN>::initializeLSTM(const Eigen::Ref<const init_buffer>& buffer)
{
  // reset hidden/cell state
  init_model_->resetHiddenCPU();

  int t = 0;
  for (t = 0; t < INITIAL_LEN - 1; t++)
  {
    init_model_->forward(buffer.col(t));
  }

  // run full model with output at end
  typename INIT_T::output_array output;
  init_model_->forward(buffer.col(t), output);

  // set the lstm initial hidden/cell to output
  lstm_->setHiddenState(output.head(HIDDEN_DIM));
  lstm_->setCellState(output.tail(HIDDEN_DIM));
}
