#ifndef DDP_MODEL_WRAPPER_H
#define DDP_MODEL_WRAPPER_H

#include "ddp_dynamics.h"
#include <boost/typeof/typeof.hpp>
#include <boost/type_traits.hpp>
#include <type_traits>

template<typename T>
struct HasAnalyticGrad
{
    template <typename U> static char Test( typeof(&U::computeGrad) ) ;
    template <typename U> static long Test(...);
    static const bool Has = sizeof(Test<T>(0)) == sizeof(char);
};

template<typename T>
bool getGrad(T* model, typename DDP_structures::Dynamics<float, T::STATE_DIM, T::CONTROL_DIM>::Jacobian &jac,
            Eigen::MatrixXf &x, Eigen::MatrixXf &u, std::true_type)
{
    // T::dfdx A;
    // T::dfdu B;
    Eigen::MatrixXf A(T::STATE_DIM, T::STATE_DIM);
    Eigen::MatrixXf B(T::STATE_DIM, T::CONTROL_DIM);
    model->computeGrad(x, u, A, B);
    jac.block(0,0, T::STATE_DIM, T::STATE_DIM) = A;
    jac.block(0, T::STATE_DIM, T::STATE_DIM, T::CONTROL_DIM) = B;
    // for (int i = 0; i < T::STATE_DIM; i++){
    //     for (int j = 0; j < T::STATE_DIM; j++){
    //         jac(i,j) = A(i,j);
    //     }
    // }
    return true;
}

template<typename T>
bool getGrad(T* model, typename DDP_structures::Dynamics<float, T::STATE_DIM, T::CONTROL_DIM>::Jacobian &jac,
            Eigen::MatrixXf &x, Eigen::MatrixXf &u, std::false_type)
{
    return false;
}

template <class DYNAMICS_T>
struct ModelWrapperDDP: public DDP_structures::Dynamics<float, DYNAMICS_T::STATE_DIM, DYNAMICS_T::CONTROL_DIM>
{
    using Scalar = float;
    using State = typename DDP_structures::Dynamics<Scalar, DYNAMICS_T::STATE_DIM, DYNAMICS_T::CONTROL_DIM>::State;
    using Control = typename DDP_structures::Dynamics<Scalar, DYNAMICS_T::STATE_DIM, DYNAMICS_T::CONTROL_DIM>::Control;
    using Jacobian = typename DDP_structures::Dynamics<Scalar, DYNAMICS_T::STATE_DIM, DYNAMICS_T::CONTROL_DIM>::Jacobian;

    Eigen::MatrixXf state;
    Eigen::MatrixXf control;

    DYNAMICS_T* model_;

    ModelWrapperDDP(DYNAMICS_T* model)
    {
        model_ = model;
    }

    State f(const Eigen::Ref<const State> &x, const Eigen::Ref<const Control> &u)
    {
        // This section is specific to the neural network implementation for the autorally
        state = x;
        control = u;
//        model_->computeKinematics(state);
//        model_->computeDynamics(state, control);
//        for (int i = 0; i < DYNAMICS_T::STATE_DIM; i++){
//            dx(i) = model_->state_der_(i);
//        }

        // Compute the state derivative xDot
        State dx; // dx can't be passed in directly at the moment
        Eigen::MatrixXf state_der(DYNAMICS_T::STATE_DIM, 1);
        model_->computeStateDeriv(state, control, state_der);
        for (int i = 0; i < DYNAMICS_T::STATE_DIM; i++) {
            dx(i) = state_der(i);
        }

        return dx;
    }

    Jacobian df(const Eigen::Ref<const State> &x, const Eigen::Ref<const Control> &u)
    {
        Jacobian j_;
        state = x;
        control = u;
        bool analyticGradComputed = getGrad(model_, j_, state, control, std::integral_constant<bool, HasAnalyticGrad<DYNAMICS_T>::Has>());
        if (!analyticGradComputed){
            j_ = DDP_structures::Dynamics<float, DYNAMICS_T::STATE_DIM, DYNAMICS_T::CONTROL_DIM>::df(x,u);
        }
        return j_;
    }

};

#endif // DDP_MODEL_WRAPPER_H
