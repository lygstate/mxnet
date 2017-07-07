/*!
*  Copyright (c) 2016 by Contributors
* \file op.h
* \brief definition of all the operators
* \author Chuntao Hong, Xin Li
*/

#ifndef CPP_PACKAGE_INCLUDE_MXNET_CPP_OP_H_
#define CPP_PACKAGE_INCLUDE_MXNET_CPP_OP_H_

#include <string>
#include <vector>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/shape.h"
#include "mxnet-cpp/op_util.h"
#include "mxnet-cpp/operator.h"
#include "dmlc/optional.h"

namespace mxnet {
namespace cpp {

/*!
 * \breif Calculate cross_entropy(data, one_hot(label))
 *
 *        From:E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\loss_binary_op.cc:12
 * \param symbol_name name of the resulting symbol
 * \param data Input data
 * \param label Input label
 * \return new symbol
 */
inline Symbol softmax_cross_entropy(const std::string& symbol_name,
                                    Symbol data,
                                    Symbol label) {
  return Operator("softmax_cross_entropy")
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Update function for Stochastic Gradient Descent (SDG) optimizer.
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * gradient
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\optimizer_op.cc:L25
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param lr Learning rate
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol sgd_update(const std::string& symbol_name,
                         Symbol weight,
                         Symbol grad,
                         mx_float lr,
                         mx_float wd = 0,
                         mx_float rescale_grad = 1,
                         mx_float clip_gradient = -1) {
  return Operator("sgd_update")
           .SetParam("lr", lr)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Momentum update function for Stochastic Gradient Descent (SDG) optimizer.
 *
 *        Momentum update has better convergence rates on neural networks. Mathematically
 *        like below:
 *
 *        .. math::
 *
 *        v_1 = \alpha * \nabla J(W_0)\\
 *        v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} + v_t
 *
 *        It updates the weights using::
 *
 *        v = momentum * v - learning_rate * gradient
 *        weight += v
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\optimizer_op.cc:L55
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param mom Momentum
 * \param lr Learning rate
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol sgd_mom_update(const std::string& symbol_name,
                             Symbol weight,
                             Symbol grad,
                             Symbol mom,
                             mx_float lr,
                             mx_float momentum = 0,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1) {
  return Operator("sgd_mom_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mom", mom)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Update function for Adam optimizer. Adam is seen as a generalization
 *        of AdaGrad.
 *
 *        Adam update consists of the following steps, where g represents gradient and m,
 *        are 1st and 2nd order moment estimates (mean and variance).
 *
 *        .. math::
 *
 *        g_t = \nabla J(W_{t-1})\\
 *        m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
 *        v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
 *        W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }
 *
 *        It updates the weights using::
 *
 *        m = beta1*m + (1-beta1)*grad
 *        v = beta2*v + (1-beta2)*(grad**2)
 *        w += - learning_rate * m / (sqrt(v) + epsilon)
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\optimizer_op.cc:L91
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param mean Moving mean
 * \param var Moving variance
 * \param lr Learning rate
 * \param beta1 The decay rate for the 1st moment estimates.
 * \param beta2 The decay rate for the 2nd moment estimates.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol adam_update(const std::string& symbol_name,
                          Symbol weight,
                          Symbol grad,
                          Symbol mean,
                          Symbol var,
                          mx_float lr,
                          mx_float beta1 = 0.9,
                          mx_float beta2 = 0.999,
                          mx_float epsilon = 1e-08,
                          mx_float wd = 0,
                          mx_float rescale_grad = 1,
                          mx_float clip_gradient = -1) {
  return Operator("adam_update")
           .SetParam("lr", lr)
           .SetParam("beta1", beta1)
           .SetParam("beta2", beta2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mean", mean)
           .SetInput("var", var)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Update function for RMSProp optimizer. The RMSProp code follows the version in
 *        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\optimizer_op.cc:L111
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param n n
 * \param lr Learning rate
 * \param gamma1 The dacay rate of momentum estimates.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param clip_weights Clip weights to the range of [-clip_weights, clip_weights] If
 *        clip_weights <= 0, weight clipping is turned off. weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmsprop_update(const std::string& symbol_name,
                             Symbol weight,
                             Symbol grad,
                             Symbol n,
                             mx_float lr,
                             mx_float gamma1 = 0.95,
                             mx_float epsilon = 1e-08,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1,
                             mx_float clip_weights = -1) {
  return Operator("rmsprop_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("n", n)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Update function for RMSPropAlex optimizer. The RMSPropAlex code follows the
 *        http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\optimizer_op.cc:L130
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param n n
 * \param g g
 * \param delta delta
 * \param lr Learning rate
 * \param gamma1 Decay rate.
 * \param gamma2 Decay rate.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param clip_weights Clip weights to the range of [-clip_weights, clip_weights] If
 *        clip_weights <= 0, weight clipping is turned off. weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmspropalex_update(const std::string& symbol_name,
                                 Symbol weight,
                                 Symbol grad,
                                 Symbol n,
                                 Symbol g,
                                 Symbol delta,
                                 mx_float lr,
                                 mx_float gamma1 = 0.95,
                                 mx_float gamma2 = 0.9,
                                 mx_float epsilon = 1e-08,
                                 mx_float wd = 0,
                                 mx_float rescale_grad = 1,
                                 mx_float clip_gradient = -1,
                                 mx_float clip_weights = -1) {
  return Operator("rmspropalex_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("gamma2", gamma2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("n", n)
           .SetInput("g", g)
           .SetInput("delta", delta)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise sum of the input arrays with broadcasting.
 *
 *        `broadcast_plus` is an alias to the function `broadcast_add`.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_add(x, y) = [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]
 *
 *        broadcast_plus(x, y) = [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L32
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_add(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise difference of the input arrays with broadcasting.
 *
 *        `broadcast_minus` is an alias to the function `broadcast_sub`.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_sub(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *        broadcast_minus(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L71
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_sub(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_sub")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise product of the input arrays with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_mul(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L104
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_mul(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_mul")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise division of the input arrays with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 6.,  6.,  6.],
 *        [ 6.,  6.,  6.]]
 *
 *        y = [[ 2.],
 *        [ 3.]]
 *
 *        broadcast_div(x, y) = [[ 3.,  3.,  3.],
 *        [ 2.,  2.,  2.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L137
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_div(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_div")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Adds arguments element-wise.
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_add(const std::string& symbol_name,
                           Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise squared value of the input.
 *
 *        .. math::
 *        square(x) = x^2
 *
 *        Example::
 *
 *        square([2, 3, 4]) = [3, 9, 16]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L269
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol square(const std::string& symbol_name,
                     Symbol data) {
  return Operator("square")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Converts each element of the input array from degrees to radians.
 *
 *        .. math::
 *        radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L507
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol radians(const std::string& symbol_name,
                      Symbol data) {
  return Operator("radians")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise square-root value of the input.
 *
 *        .. math::
 *        \textrm{sqrt}(x) = \sqrt{x}
 *
 *        Example::
 *
 *        sqrt([4, 9, 16]) = [2, 3, 4]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L287
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sqrt(const std::string& symbol_name,
                   Symbol data) {
  return Operator("sqrt")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the hyperbolic cosine  of the input array, computed element-wise.
 *
 *        .. math::
 *        cosh(x) = 0.5\times(exp(x) + exp(-x))
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L535
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol cosh(const std::string& symbol_name,
                   Symbol data) {
  return Operator("cosh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise inverse square-root value of the input.
 *
 *        .. math::
 *        rsqrt(x) = 1/\sqrt{x}
 *
 *        Example::
 *
 *        rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L305
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol rsqrt(const std::string& symbol_name,
                    Symbol data) {
  return Operator("rsqrt")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the hyperbolic sine of the input array, computed element-wise.
 *
 *        .. math::
 *        sinh(x) = 0.5\times(exp(x) - exp(-x))
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L521
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sinh(const std::string& symbol_name,
                   Symbol data) {
  return Operator("sinh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise exponential value of the input.
 *
 *        .. math::
 *        exp(x) = e^x \approx 2.718^x
 *
 *        Example::
 *
 *        exp([0, 1, 2]) = [inf, 1, 0.707]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L324
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol exp(const std::string& symbol_name,
                  Symbol data) {
  return Operator("exp")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the hyperbolic tangent of the input array, computed element-wise.
 *
 *        .. math::
 *        tanh(x) = sinh(x) / cosh(x)
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L549
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol tanh(const std::string& symbol_name,
                   Symbol data) {
  return Operator("tanh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise Natural logarithmic value of the input.
 *
 *        The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L334
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log(const std::string& symbol_name,
                  Symbol data) {
  return Operator("log")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise Base-10 logarithmic value of the input.
 *
 *        ``10**log10(x) = x``
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L344
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log10(const std::string& symbol_name,
                    Symbol data) {
  return Operator("log10")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the element-wise inverse hyperbolic sine of the input array, computed
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L559
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arcsinh(const std::string& symbol_name,
                      Symbol data) {
  return Operator("arcsinh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise Base-2 logarithmic value of the input.
 *
 *        ``2**log2(x) = x``
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L354
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log2(const std::string& symbol_name,
                   Symbol data) {
  return Operator("log2")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes rectified linear.
 *
 *        .. math::
 *        max(features, 0)
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L18
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol relu(const std::string& symbol_name,
                   Symbol data) {
  return Operator("relu")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the element-wise inverse hyperbolic cosine of the input array, computed
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L569
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arccosh(const std::string& symbol_name,
                      Symbol data) {
  return Operator("arccosh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise ``log(1 + x)`` value of the input.
 *
 *        This function is more accurate than ``log(1 + x)``  for small ``x`` so that
 *        :math:`1+x\approx 1`
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L384
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log1p(const std::string& symbol_name,
                    Symbol data) {
  return Operator("log1p")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes sigmoid of x element-wise.
 *
 *        .. math::
 *        y = 1 / (1 + exp(-x))
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L36
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sigmoid(const std::string& symbol_name,
                      Symbol data) {
  return Operator("sigmoid")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the element-wise inverse hyperbolic tangent of the input array,
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L579
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arctanh(const std::string& symbol_name,
                      Symbol data) {
  return Operator("arctanh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns ``exp(x) - 1`` computed element-wise on the input.
 *
 *        This function provides greater precision than ``exp(x) - 1`` for small values
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L397
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol expm1(const std::string& symbol_name,
                    Symbol data) {
  return Operator("expm1")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns the gamma function (extension of the factorial function to the reals) ,
 *
 *        From:E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:589
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol gamma(const std::string& symbol_name,
                    Symbol data) {
  return Operator("gamma")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes the element-wise sine of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L370
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sin(const std::string& symbol_name,
                  Symbol data) {
  return Operator("sin")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Stops gradient computation.
 *
 *        Stops the accumulated gradient of the inputs from flowing through this operator
 *        in the backward direction. In other words, this operator prevents the
 *        of its inputs to be taken into account for computing gradients.
 *
 *        Example::
 *
 *        v1 = [1, 2]
 *        v2 = [0, 1]
 *        a = Variable('a')
 *        b = Variable('b')
 *        b_stop_grad = stop_gradient(3 * b)
 *        loss = MakeLoss(b_stop_grad + a)
 *
 *        executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
 *        executor.forward(is_train=True, a=v1, b=v2)
 *        executor.outputs
 *        [ 1.  5.]
 *
 *        executor.backward()
 *        executor.grad_arrays
 *        [ 0.  0.]
 *        [ 1.  1.]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L91
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol BlockGrad(const std::string& symbol_name,
                        Symbol data) {
  return Operator("BlockGrad")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Stops gradient computation.
 *        .. note:: ``make_loss`` is deprecated, use ``MakeLoss``.
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L98
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol make_loss(const std::string& symbol_name,
                        Symbol data) {
  return Operator("make_loss")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise log of the absolute value of the gamma function of the
 *
 *        From:E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:599
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol gammaln(const std::string& symbol_name,
                      Symbol data) {
  return Operator("gammaln")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes the element-wise cosine of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L413
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol cos(const std::string& symbol_name,
                  Symbol data) {
  return Operator("cos")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Output data type.
 */
enum class CastDtype {
  float16 = 0,
  float32 = 1,
  float64 = 2,
  int32 = 3,
  uint8 = 4
};

/*!
 * \breif Casts all elements of the input to a new type.
 *
 *        .. note:: ``Cast`` is deprecated. Use ``cast`` instead.
 *
 *        Example::
 *
 *        cast([0.9, 1.3], dtype='int32') = [0, 1]
 *        cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
 *        cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L145
 * \param symbol_name name of the resulting symbol
 * \param data The input.
 * \param dtype Output data type.
 * \return new symbol
 */
inline Symbol Cast(const std::string& symbol_name,
                   Symbol data,
                   CastDtype dtype) {
  static const char *CastDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "uint8"
  };
  return Operator("Cast")
           .SetParam("dtype", CastDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Negate src
 *
 *        From:E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:164
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol negative(const std::string& symbol_name,
                       Symbol data) {
  return Operator("negative")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes the element-wise tangent of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        tan([0, \pi/4, \pi/2]) = [0, 1, -inf]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L429
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol tan(const std::string& symbol_name,
                  Symbol data) {
  return Operator("tan")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise absolute value of the input.
 *
 *        Example::
 *
 *        abs([-2, 0, 3]) = [2, 0, 3]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L176
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol abs(const std::string& symbol_name,
                  Symbol data) {
  return Operator("abs")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise inverse sine of the input array.
 *
 *        The input should be in the range `[-1, 1]`.
 *        The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].
 *
 *        .. math::
 *        arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L446
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arcsin(const std::string& symbol_name,
                     Symbol data) {
  return Operator("arcsin")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise sign of the input.
 *
 *        Example::
 *
 *        sign([-2, 0, 3]) = [-1, 0, 1]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L191
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sign(const std::string& symbol_name,
                   Symbol data) {
  return Operator("sign")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise inverse cosine of the input array.
 *
 *        The input should be in range `[-1, 1]`.
 *        The output is in the closed interval :math:`[0, \pi]`
 *
 *        .. math::
 *        arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L463
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arccos(const std::string& symbol_name,
                     Symbol data) {
  return Operator("arccos")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise rounded value to the nearest integer of the input.
 *
 *        Example::
 *
 *        round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L206
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol round(const std::string& symbol_name,
                    Symbol data) {
  return Operator("round")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise ceiling of the input.
 *
 *        Example::
 *
 *        ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L233
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol ceil(const std::string& symbol_name,
                   Symbol data) {
  return Operator("ceil")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise inverse tangent of the input array.
 *
 *        The output is in the closed interval :math:`[-\pi/2, \pi/2]`
 *
 *        .. math::
 *        arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L479
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arctan(const std::string& symbol_name,
                     Symbol data) {
  return Operator("arctan")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise floor of the input.
 *
 *        Example::
 *
 *        floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L244
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol floor(const std::string& symbol_name,
                    Symbol data) {
  return Operator("floor")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise rounded value to the nearest integer of the input.
 *
 *        .. note::
 *        - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.
 *        - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.
 *
 *        Example::
 *
 *        rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L222
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol rint(const std::string& symbol_name,
                   Symbol data) {
  return Operator("rint")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Converts each element of the input array from radians to degrees.
 *
 *        .. math::
 *        degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L493
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol degrees(const std::string& symbol_name,
                      Symbol data) {
  return Operator("degrees")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Returns element-wise rounded value to the nearest integer towards zero of the
 *
 *        Example::
 *
 *        fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L255
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol fix(const std::string& symbol_name,
                  Symbol data) {
  return Operator("fix")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Slices a contiguous region of the array.
 *
 *        .. note:: ``crop`` is deprecated. Use ``slice`` instead.
 *
 *        This function returns a sliced continous region of the array between the
 *        by `begin` and `end`.
 *
 *        For an input array of `n` dimensions, slice operation with ``begin=(b_0,
 *        and ``end=(e_1, e_2, ... e_n)`` indices will result in an array with the shape
 *        ``(e_1-b_0, ..., e_n-b_n-1)``.
 *
 *        The resulting array's *k*-th dimension contains elements
 *        from the *k*-th dimension of the input array with the open range ``[b_k, e_k)``.
 *
 *        Example::
 *
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
 *        [ 6.,  7.,  8.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L244
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param begin starting indices for the slice operation, supports negative indices.
 * \param end ending indices for the slice operation, supports negative indices.
 * \return new symbol
 */
inline Symbol slice(const std::string& symbol_name,
                    Symbol data,
                    Shape begin,
                    Shape end) {
  return Operator("slice")
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Slices along a given axis.
 *
 *        Returns an array slice along a given `axis` starting from the `begin` index
 *        to the `end` index.
 *
 *        Examples::
 *
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],
 *        [  5.,   6.],
 *        [  9.,  10.]]
 *
 *        slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],
 *        [  6.,   7.],
 *        [ 10.,  11.]]
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L324
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Axis along which to be sliced, supports negative indexes.
 * \param begin The beginning index along the axis to be sliced,  supports negative
 * \param end The ending index along the axis to be sliced,  supports negative indexes.
 * \return new symbol
 */
inline Symbol slice_axis(const std::string& symbol_name,
                         Symbol data,
                         int axis,
                         int begin,
                         dmlc::optional<int> end) {
  return Operator("slice_axis")
           .SetParam("axis", axis)
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Dot product of two arrays.
 *
 *        ``dot``'s behavior depends on the input array dimensions:
 *
 *        - 1-D arrays: inner product of vectors
 *        - 2-D arrays: matrix multiplication
 *        - N-D arrays: a sum product over the last axis of the first input and the first
 *        axis of the second input
 *
 *        For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape
 *        result array will have shape `(n,m,r,s)`. It is computed by::
 *
 *        dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L357
 * \param symbol_name name of the resulting symbol
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \return new symbol
 */
inline Symbol dot(const std::string& symbol_name,
                  Symbol lhs,
                  Symbol rhs,
                  bool transpose_a = false,
                  bool transpose_b = false) {
  return Operator("dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Batchwise dot product.
 *
 *        ``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
 *        ``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.
 *
 *        For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape
 *        `(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,
 *        which is computed by::
 *
 *        batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L393
 * \param symbol_name name of the resulting symbol
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \return new symbol
 */
inline Symbol batch_dot(const std::string& symbol_name,
                        Symbol lhs,
                        Symbol rhs,
                        bool transpose_a = false,
                        bool transpose_b = false) {
  return Operator("batch_dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Clips (limits) the values in an array.
 *
 *        Given an interval, values outside the interval are clipped to the interval
 *        Clipping ``x`` between `a_min` and `a_x` would be::
 *
 *        clip(x, a_min, a_max) = max(min(x, a_max), a_min))
 *
 *        Example::
 *
 *        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 *
 *        clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L438
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \param a_min Minimum value
 * \param a_max Maximum value
 * \return new symbol
 */
inline Symbol clip(const std::string& symbol_name,
                   Symbol data,
                   mx_float a_min,
                   mx_float a_max) {
  return Operator("clip")
           .SetParam("a_min", a_min)
           .SetParam("a_max", a_max)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Repeats elements of an array.
 *
 *        By default, ``repeat`` flattens the input array into 1-D and then repeats the
 *        elements::
 *
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *
 *        repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]
 *
 *        The parameter ``axis`` specifies the axis along which to perform repeat::
 *
 *        repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],
 *        [ 3.,  3.,  4.,  4.]]
 *
 *        repeat(x, repeats=2, axis=0) = [[ 1.,  2.],
 *        [ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 3.,  4.]]
 *
 *        repeat(x, repeats=2, axis=-1) = [[ 1.,  1.,  2.,  2.],
 *        [ 3.,  3.,  4.,  4.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L480
 * \param symbol_name name of the resulting symbol
 * \param data Input data array
 * \param repeats The number of repetitions for each element.
 * \param axis The axis along which to repeat values. The negative numbers are
 *        interpreted counting from the backward. By default, use the flattened input
 * \return new symbol
 */
inline Symbol repeat(const std::string& symbol_name,
                     Symbol data,
                     int repeats,
                     dmlc::optional<int> axis = dmlc::optional<int>()) {
  return Operator("repeat")
           .SetParam("repeats", repeats)
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Repeats the whole array multiple times.
 *
 *        If ``reps`` has length *d*, and input array has dimension of *n*. There are
 *        there cases:
 *
 *        - **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::
 *
 *        x = [[1, 2],
 *        [3, 4]]
 *
 *        tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]
 *
 *        - **n>d**. ``reps`` is promoted to length *n* by pre-pending 1's to it. Thus for
 *        an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::
 *
 *
 *        tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.]]
 *
 *        - **n<d**. The input is promoted to be d-dimensional by prepending new axes. So
 *        shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::
 *
 *        tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]],
 *
 *        [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]]
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L537
 * \param symbol_name name of the resulting symbol
 * \param data Input data array
 * \param reps The number of times for repeating the tensor a. If reps has length d, the
 *        result will have dimension of max(d, a.ndim); If a.ndim < d, a is promoted to
 *        be d-dimensional by prepending new axes. If a.ndim > d, reps is promoted to
 * \return new symbol
 */
inline Symbol tile(const std::string& symbol_name,
                   Symbol data,
                   Shape reps) {
  return Operator("tile")
           .SetParam("reps", reps)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Reverses the order of elements along given axis while preserving array shape.
 *
 *        Note: reverse and flip are equivalent. We use reverse in the following examples.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.,  3.,  4.],
 *        [ 5.,  6.,  7.,  8.,  9.]]
 *
 *        reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],
 *        [ 0.,  1.,  2.,  3.,  4.]]
 *
 *        reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],
 *        [ 9.,  8.,  7.,  6.,  5.]]
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L574
 * \param symbol_name name of the resulting symbol
 * \param data Input data array
 * \param axis The axis which to reverse elements.
 * \return new symbol
 */
inline Symbol reverse(const std::string& symbol_name,
                      Symbol data,
                      Shape axis) {
  return Operator("reverse")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Reshapes the input array.
 *
 *        .. note:: ``Reshape`` is deprecated, use ``reshape``
 *
 *        Given an array and a shape, this function returns a copy of the array in the
 *        The shape is a tuple of integers such as (2,3,4).The size of the new shape
 *
 *        Example::
 *
 *        reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]
 *
 *        Some dimensions of the shape can take special values from the set {0, -1, -2,
 *
 *        - ``0``  copy this dimension from the input to the output shape.
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
 *        - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)
 *
 *        - ``-1`` infers the dimension of the output shape by using the remainder of the
 *        keeping the size of the new array same as that of the input array.
 *        At most one dimension of shape can be -1.
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
 *        - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
 *        - input shape = (2,3,4), shape=(-1,), output shape = (24,)
 *
 *        - ``-2`` copy all/remainder of the input dimensions to the output shape.
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
 *        - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
 *        - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)
 *
 *        - ``-3`` use the product of two consecutive dimensions of the input shape as
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
 *        - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
 *        - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
 *        - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)
 *
 *        - ``-4`` split one dimension of the input into two dimensions passed subsequent
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
 *        - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)
 *
 *        If the argument `reverse` is set to 1, then the special values are inferred
 *
 *        Example::
 *
 *        - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape
 *        - with reverse=1, output shape will be (50,4).
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L87
 * \param symbol_name name of the resulting symbol
 * \param data Input data to reshape.
 * \param shape The target shape
 * \param reverse If true then the special values are inferred from right to left
 * \param target_shape (Deprecated! Use ``shape`` instead.) Target new shape. One and
 * \param keep_highest (Deprecated! Use ``shape`` instead.) Whether keep the highest dim
 *        unchanged.If set to true, then the first dim in target_shape is ignored,and
 * \return new symbol
 */
inline Symbol Reshape(const std::string& symbol_name,
                      Symbol data,
                      Shape shape = Shape(),
                      bool reverse = false,
                      Shape target_shape = Shape(0,0),
                      bool keep_highest = false) {
  return Operator("Reshape")
           .SetParam("shape", shape)
           .SetParam("reverse", reverse)
           .SetParam("target_shape", target_shape)
           .SetParam("keep_highest", keep_highest)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Flattens the input array into a 2-D array by collapsing the higher dimensions.
 *
 *        .. note:: `Flatten` is deprecated. Use `flatten` instead.
 *
 *        For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation
 *        the input array into an output array of shape ``(d1, d2*...*dk)``.
 *
 *        Example::
 *
 *        x = [[
 *        [1,2,3],
 *        [4,5,6],
 *        [7,8,9]
 *        ],
 *        [    [1,2,3],
 *        [4,5,6],
 *        [7,8,9]
 *        ]],
 *
 *        flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
 *        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L127
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \return new symbol
 */
inline Symbol Flatten(const std::string& symbol_name,
                      Symbol data) {
  return Operator("Flatten")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Permutes the dimensions of an array.
 *
 *        Examples::
 *
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *
 *        transpose(x) = [[ 1.,  3.],
 *        [ 2.,  4.]]
 *
 *        x = [[[ 1.,  2.],
 *        [ 3.,  4.]],
 *
 *        [[ 5.,  6.],
 *        [ 7.,  8.]]]
 *
 *        transpose(x) = [[[ 1.,  5.],
 *        [ 3.,  7.]],
 *
 *        [[ 2.,  6.],
 *        [ 4.,  8.]]]
 *
 *        transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],
 *        [ 5.,  6.]],
 *
 *        [[ 3.,  4.],
 *        [ 7.,  8.]]]
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L168
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axes Target axis order. By default the axes will be inverted.
 * \return new symbol
 */
inline Symbol transpose(const std::string& symbol_name,
                        Symbol data,
                        Shape axes = Shape()) {
  return Operator("transpose")
           .SetParam("axes", axes)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Inserts a new axis of size 1 into the array shape
 *
 *        For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``
 *        will return a new array with shape ``(2,1,3,4)``.
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L204
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Position (amongst axes) where new axis is to be inserted.
 * \return new symbol
 */
inline Symbol expand_dims(const std::string& symbol_name,
                          Symbol data,
                          uint32_t axis) {
  return Operator("expand_dims")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Batch normalization.
 *
 *        Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
 *        well as offset ``beta``.
 *
 *        Assume the input has more than one dimension and we normalize along axis 1.
 *        We first compute the mean and variance along this axis:
 *
 *        .. math::
 *
 *        data\_mean[i] = mean(data[:,i,:,...]) \\
 *        data\_var[i] = var(data[:,i,:,...])
 *
 *        Then compute the normalized output, which has the same shape as input, as
 *
 *        .. math::
 *
 *        out[:,i,:,...] = \frac{data[:,i,:,...] -
 *
 *        Both *mean* and *var* returns a scalar by treating the input as a vector.
 *
 *        Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
 *        have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both
 *        ``data_var`` as well, which are needed for the backward pass.
 *
 *        Besides the inputs and the outputs, this operator accepts two auxiliary
 *        states, ``moving_mean`` and ``moving_var``, which are *k*-length
 *        vectors. They are global statistics for the whole dataset, which are updated
 *        by::
 *
 *        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
 *        moving_var = moving_var * momentum + data_var * (1 - momentum)
 *
 *        If ``use_global_stats`` is set to be true, then ``moving_mean`` and
 *        ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
 *        the output. It is often used during inference.
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is
 *        then set ``gamma`` to 1 and its gradient to 0.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data Input data to batch normalization
 * \param gamma gamma array
 * \param beta beta array
 * \param eps Epsilon to prevent div 0. Must be bigger than CUDNN_BN_MIN_EPSILON defined
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of local
 * \param output_mean_var Output All,normal mean and var
 * \return new symbol
 */
inline Symbol BatchNorm(const std::string& symbol_name,
                        Symbol data,
                        Symbol gamma,
                        Symbol beta,
                        mx_float eps = 0.001,
                        mx_float momentum = 0.9,
                        bool fix_gamma = true,
                        bool use_global_stats = false,
                        bool output_mean_var = false) {
  return Operator("BatchNorm")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetParam("output_mean_var", output_mean_var)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Joins input arrays along a given axis.
 *
 *        .. note:: `Concat` is deprecated. Use `concat` instead.
 *
 *        The dimensions of the input arrays should be the same except the axis along
 *        which they will concatenated.
 *        The dimension of the output array along the concatenated axis will be equal
 *        to the sum of the corresponding dimensions of the input arrays.
 *
 *        Example::
 *
 *        x = [[1,1],[2,2]]
 *        y = [[3,3],[4,4],[5,5]]
 *        z = [[6,6], [7,7],[8,8]]
 *
 *        concat(x,y,z,dim=0) = [[ 1.,  1.],
 *        [ 2.,  2.],
 *        [ 3.,  3.],
 *        [ 4.,  4.],
 *        [ 5.,  5.],
 *        [ 6.,  6.],
 *        [ 7.,  7.],
 *        [ 8.,  8.]]
 *
 *        Note that you cannot concat x,y,z along dimension 1 since dimension
 *        0 is not the same for all the input arrays.
 *
 *        concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
 *        [ 4.,  4.,  7.,  7.],
 *        [ 5.,  5.,  8.,  8.]]
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data List of arrays to concatenate
 * \param num_args Number of inputs to be concated.
 * \param dim the dimension to be concated.
 * \return new symbol
 */
inline Symbol Concat(const std::string& symbol_name,
                     const std::vector<Symbol>& data,
                     int num_args,
                     int dim = 1) {
  return Operator("Concat")
           .SetParam("num_args", num_args)
           .SetParam("dim", dim)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply a sparse regularization to the output a sigmoid activation function.
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \param sparseness_target The sparseness target
 * \param penalty The tradeoff parameter for the sparseness penalty
 * \param momentum The momentum for running average
 * \return new symbol
 */
inline Symbol IdentityAttachKLSparseReg(const std::string& symbol_name,
                                        Symbol data,
                                        mx_float sparseness_target = 0.1,
                                        mx_float penalty = 0.001,
                                        mx_float momentum = 0.9) {
  return Operator("IdentityAttachKLSparseReg")
           .SetParam("sparseness_target", sparseness_target)
           .SetParam("penalty", penalty)
           .SetParam("momentum", momentum)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Activation function to be applied.
 */
enum class LeakyReLUActType {
  elu = 0,
  leaky = 1,
  prelu = 2,
  rrelu = 3
};

/*!
 * \breif Applies Leaky rectified linear unit activation element-wise to the input.
 *
 *        Leaky ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope`
 *        when the input is negative and has a slope of one when input is positive.
 *
 *        The following modified ReLU Activation functions are supported:
 *
 *        - *elu*: Exponential Linear Unit. `y = x > 0 ? x : slope * (exp(x)-1)`
 *        - *leaky*: Leaky ReLU. `y = x > 0 ? x : slope * x`
 *        - *prelu*: Parametric ReLU. This is same as *leaky* except that `slope` is
 *        - *rrelu*: Randomized ReLU. same as *leaky* but the `slope` is uniformly and
 *        *[lower_bound, upper_bound)* for training, while fixed to be
 *        *(lower_bound+upper_bound)/2* for inference.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data Input data to activation function.
 * \param act_type Activation function to be applied.
 * \param slope Init slope for the activation. (For leaky and elu only)
 * \param lower_bound Lower bound of random slope. (For rrelu only)
 * \param upper_bound Upper bound of random slope. (For rrelu only)
 * \return new symbol
 */
inline Symbol LeakyReLU(const std::string& symbol_name,
                        Symbol data,
                        LeakyReLUActType act_type = LeakyReLUActType::leaky,
                        mx_float slope = 0.25,
                        mx_float lower_bound = 0.125,
                        mx_float upper_bound = 0.334) {
  static const char *LeakyReLUActTypeValues[] = {
    "elu",
    "leaky",
    "prelu",
    "rrelu"
  };
  return Operator("LeakyReLU")
           .SetParam("act_type", LeakyReLUActTypeValues[int(act_type)])
           .SetParam("slope", slope)
           .SetParam("lower_bound", lower_bound)
           .SetParam("upper_bound", upper_bound)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Padding type to use. "constant" pads with `constant_value` and "edge" pads
 */
enum class PadMode {
  constant = 0,
  edge = 1
};

/*!
 * \breif Pads an input array with a constant or edge values of the array.
 *
 *        .. note:: `Pad` is deprecated. Use `pad` instead.
 *
 *        .. note:: Current implementation only supports 4D and 5D input arrays with
 *        only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.
 *
 *        This operation pads an input array with either a `constant_value` or edge values
 *        along each axis of the input array. The amount of padding is specified by
 *
 *        `pad_width` is a tuple of integer padding widths for each axis of the format
 *        ``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of
 *        where ``N`` is the number of dimensions of the array.
 *
 *        For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates
 *        to add before and after the elements of the array along dimension ``N``.
 *        The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,
 *        ``after_2`` must be 0.
 *
 *        Example::
 *
 *        x = [[[[  1.   2.   3.]
 *        [  4.   5.   6.]]
 *
 *        [[  7.   8.   9.]
 *        [ 10.  11.  12.]]]
 *
 *
 *        [[[ 11.  12.  13.]
 *        [ 14.  15.  16.]]
 *
 *        [[ 17.  18.  19.]
 *        [ 20.  21.  22.]]]]
 *
 *        pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =
 *
 *        [[[[  1.   1.   2.   3.   3.]
 *        [  1.   1.   2.   3.   3.]
 *        [  4.   4.   5.   6.   6.]
 *        [  4.   4.   5.   6.   6.]]
 *
 *        [[  7.   7.   8.   9.   9.]
 *        [  7.   7.   8.   9.   9.]
 *        [ 10.  10.  11.  12.  12.]
 *        [ 10.  10.  11.  12.  12.]]]
 *
 *
 *        [[[ 11.  11.  12.  13.  13.]
 *        [ 11.  11.  12.  13.  13.]
 *        [ 14.  14.  15.  16.  16.]
 *        [ 14.  14.  15.  16.  16.]]
 *
 *        [[ 17.  17.  18.  19.  19.]
 *        [ 17.  17.  18.  19.  19.]
 *        [ 20.  20.  21.  22.  22.]
 *        [ 20.  20.  21.  22.  22.]]]]
 *
 *        pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,2,2,1,1)) =
 *
 *        [[[[  0.   0.   0.   0.   0.]
 *        [  0.   1.   2.   3.   0.]
 *        [  0.   4.   5.   6.   0.]
 *        [  0.   0.   0.   0.   0.]]
 *
 *        [[  0.   0.   0.   0.   0.]
 *        [  0.   7.   8.   9.   0.]
 *        [  0.  10.  11.  12.   0.]
 *        [  0.   0.   0.   0.   0.]]]
 *
 *
 *        [[[  0.   0.   0.   0.   0.]
 *        [  0.  11.  12.  13.   0.]
 *        [  0.  14.  15.  16.   0.]
 *        [  0.   0.   0.   0.   0.]]
 *
 *        [[  0.   0.   0.   0.   0.]
 *        [  0.  17.  18.  19.   0.]
 *        [  0.  20.  21.  22.   0.]
 *        [  0.   0.   0.   0.   0.]]]]
 *
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data An n-dimensional input array.
 * \param mode Padding type to use. "constant" pads with `constant_value` and "edge" pads
 * \param pad_width Widths of the padding regions applied to the edges of each axis. It
 *        is a tuple of integer padding widths for each axis of the format ``(before_1,
 *        after_1, ... , before_N, after_N)``. It should be of length ``2*N`` where ``N``
 *        is the number of dimensions of the array.This is equivalent to pad_width in
 * \param constant_value The value used for padding when `mode` is "constant".
 * \return new symbol
 */
inline Symbol Pad(const std::string& symbol_name,
                  Symbol data,
                  PadMode mode,
                  Shape pad_width,
                  double constant_value = 0) {
  static const char *PadModeValues[] = {
    "constant",
    "edge"
  };
  return Operator("Pad")
           .SetParam("mode", PadModeValues[int(mode)])
           .SetParam("pad_width", pad_width)
           .SetParam("constant_value", constant_value)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Splits an array along a particular axis into multiple sub-arrays.
 *
 *        .. note:: ``SliceChannel`` is depreacted. Use ``split`` instead.
 *
 *        **Note** that `num_outputs` should evenly divide the length of the axis
 *        along which to split the array.
 *
 *        Example::
 *
 *        x  = [[[ 1.]
 *        [ 2.]]
 *        [[ 3.]
 *        [ 4.]]
 *        [[ 5.]
 *        [ 6.]]]
 *        x.shape = (3, 2, 1)
 *
 *        y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)
 *        y = [[[ 1.]]
 *        [[ 3.]]
 *        [[ 5.]]]
 *
 *        [[[ 2.]]
 *        [[ 4.]]
 *        [[ 6.]]]
 *
 *        y[0].shape = (3, 1, 1)
 *
 *        z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)
 *        z = [[[ 1.]
 *        [ 2.]]]
 *
 *        [[[ 3.]
 *        [ 4.]]]
 *
 *        [[[ 5.]
 *        [ 6.]]]
 *
 *        z[0].shape = (1, 2, 1)
 *
 *        `squeeze_axis=1` removes the axis with length 1 from the shapes of the output
 *        **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
 *        along the `axis` which it is split.
 *        Also `squeeze_axis` can be set to true only if ``input.shape[axis] ==
 *
 *        z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with
 *        z = [[ 1.]
 *        [ 2.]]
 *
 *        [[ 3.]
 *        [ 4.]]
 *
 *        [[ 5.]
 *        [ 6.]]
 *        z[0].shape = (2 ,1 )
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\slice_channel.cc:L86
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param num_outputs Number of splits. Note that this should evenly divide the length of
 * \param axis Axis along which to split.
 * \param squeeze_axis If true, Removes the axis with length 1 from the shapes of the
 *        output arrays. **Note** that setting `squeeze_axis` to ``true`` removes axis
 *        with length 1 only along the `axis` which it is split. Also `squeeze_axis` can
 * \return new symbol
 */
inline Symbol SliceChannel(const std::string& symbol_name,
                           Symbol data,
                           int num_outputs,
                           int axis = 1,
                           bool squeeze_axis = false) {
  return Operator("SliceChannel")
           .SetParam("num_outputs", num_outputs)
           .SetParam("axis", axis)
           .SetParam("squeeze_axis", squeeze_axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Interchanges two axes of an array.
 *
 *        Examples::
 *
 *        x = [[1, 2, 3]])
 *        swapaxes(x, 0, 1) = [[ 1],
 *        [ 2],
 *        [ 3]]
 *
 *        x = [[[ 0, 1],
 *        [ 2, 3]],
 *        [[ 4, 5],
 *        [ 6, 7]]]  // (2,2,2) array
 *
 *        swapaxes(x, 0, 2) = [[[ 0, 4],
 *        [ 2, 6]],
 *        [[ 1, 5],
 *        [ 3, 7]]]
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \param dim1 the first axis to be swapped.
 * \param dim2 the second axis to be swapped.
 * \return new symbol
 */
inline Symbol SwapAxis(const std::string& symbol_name,
                       Symbol data,
                       uint32_t dim1 = 0,
                       uint32_t dim2 = 0) {
  return Operator("SwapAxis")
           .SetParam("dim1", dim1)
           .SetParam("dim2", dim2)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif upsampling method
 */
enum class UpSamplingSampleType {
  bilinear = 0,
  nearest = 1
};

/*! \breif How to handle multiple input. concat means concatenate upsampled images along
 *        the channel dimension. sum means add all images together, only available for
 */
enum class UpSamplingMultiInputMode {
  concat = 0,
  sum = 1
};

/*!
 * \breif Performs nearest neighbor/bilinear up sampling to inputs
 * \param symbol_name name of the resulting symbol
 * \param data Array of tensors to upsample
 * \param scale Up sampling scale
 * \param sample_type upsampling method
 * \param num_args Number of inputs to be upsampled. For nearest neighbor upsampling,
 *        this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other
 *        inputs will be upsampled to thesame size. For bilinear upsampling this must be
 * \param num_filter Input filter. Only used by bilinear sample_type.
 * \param multi_input_mode How to handle multiple input. concat means concatenate
 *        upsampled images along the channel dimension. sum means add all images
 * \param workspace Tmp workspace for deconvolution (MB)
 * \return new symbol
 */
inline Symbol UpSampling(const std::string& symbol_name,
                         const std::vector<Symbol>& data,
                         uint32_t scale,
                         UpSamplingSampleType sample_type,
                         int num_args,
                         uint32_t num_filter = 0,
                         UpSamplingMultiInputMode multi_input_mode = UpSamplingMultiInputMode::concat,
                         uint64_t workspace = 512) {
  static const char *UpSamplingSampleTypeValues[] = {
    "bilinear",
    "nearest"
  };
  static const char *UpSamplingMultiInputModeValues[] = {
    "concat",
    "sum"
  };
  return Operator("UpSampling")
           .SetParam("scale", scale)
           .SetParam("sample_type", UpSamplingSampleTypeValues[int(sample_type)])
           .SetParam("num_args", num_args)
           .SetParam("num_filter", num_filter)
           .SetParam("multi_input_mode", UpSamplingMultiInputModeValues[int(multi_input_mode)])
           .SetParam("workspace", workspace)
(data)
           .CreateSymbol(symbol_name);
}

/*! \breif Activation function to be applied.
 */
enum class ActivationActType {
  relu = 0,
  sigmoid = 1,
  softrelu = 2,
  tanh = 3
};

/*!
 * \breif Applies an activation function element-wise to the input.
 *
 *        The following activation functions are supported:
 *
 *        - `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
 *        - `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
 *        - `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) +
 *        - `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data Input array to activation function.
 * \param act_type Activation function to be applied.
 * \return new symbol
 */
inline Symbol Activation(const std::string& symbol_name,
                         Symbol data,
                         ActivationActType act_type) {
  static const char *ActivationActTypeValues[] = {
    "relu",
    "sigmoid",
    "softrelu",
    "tanh"
  };
  return Operator("Activation")
           .SetParam("act_type", ActivationActTypeValues[int(act_type)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Applies bilinear sampling to input feature map, which is the key of "[NIPS2015]
 *        output[batch, channel, y_dst, x_dst] = G(data[batch, channel, y_src, x_src)
 *        x_dst, y_dst enumerate all spatial locations in output
 *        x_src = grid[batch, 0, y_dst, x_dst]
 *        y_src = grid[batch, 1, y_dst, x_dst]
 *        G() denotes the bilinear interpolation kernel
 *        The out-boundary points will be padded as zeros. (The boundary is defined to be
 *        The shape of output will be (data.shape[0], data.shape[1], grid.shape[2],
 *        The operator assumes that grid has been nomalized. If you want to design a
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the BilinearsamplerOp.
 * \param grid Input grid to the BilinearsamplerOp.grid has two channels: x_src, y_src
 * \return new symbol
 */
inline Symbol BilinearSampler(const std::string& symbol_name,
                              Symbol data,
                              Symbol grid) {
  return Operator("BilinearSampler")
           .SetInput("data", data)
           .SetInput("grid", grid)
           .CreateSymbol(symbol_name);
}

/*! \breif Whether to pick convolution algo by running performance test.
 */
enum class ConvolutionCudnnTune {
  None = 0,
  fastest = 1,
  limited_workspace = 2,
  off = 3
};

/*! \breif Set layout for input, output and weight. Empty for
 *        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
 */
enum class ConvolutionLayout {
  None = 0,
  NCDHW = 1,
  NCHW = 2,
  NCW = 3,
  NDHWC = 4,
  NHWC = 5
};

/*!
 * \breif Compute *N*-D convolution on *(N+2)*-D input.
 *
 *        In the 2-D convolution, given input data with shape *(batch_size,
 *        channel, height, width)*, the output is computed by
 *
 *        .. math::
 *
 *        out[n,i,:,:] = bias[i] + \sum_{j=0}^{num\_filter} data[n,j,:,:] \star
 *        weight[i,j,:,:]
 *
 *        where :math:`\star` is the 2-D cross-correlation operator.
 *
 *        For general 2-D convolution, the shapes are
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*.
 *
 *        Define::
 *
 *        f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1
 *
 *        then we have::
 *
 *        out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
 *        out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *        The default data ``layout`` is *NCHW*, namely *(batch_size, channle, height,
 *        width)*. We can choose other layouts such as *NHWC*.
 *
 *        If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
 *        evenly into *g* parts along the channel axis, and also evenly split ``weight``
 *        along the first dimension. Next compute the convolution on the *i*-th part of
 *        the data with the *i*-th weight part. The output is obtained by concating all
 *        the *g* results.
 *
 *        1-D convolution does not have *height* dimension but only *width* in space.
 *
 *        - **data**: *(batch_size, channel, width)*
 *        - **weight**: *(num_filter, channel, kernel[0])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_width)*.
 *
 *        3-D convolution adds an additional *depth* dimension besides *height* and
 *        *width*. The shapes are
 *
 *        - **data**: *(batch_size, channel, depth, height, width)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.
 *
 *        Both ``weight`` and ``bias`` are learnable parameters.
 *
 *        There are other options to tune the performance.
 *
 *        - **cudnn_tune**: enable this option leads to higher startup time but may give
 *        faster speed. Options are
 *
 *        - **off**: no tuning
 *        - **limited_workspace**:run test and pick the fastest algorithm that doesn't
 *        exceed workspace limit.
 *        - **fastest**: pick the fastest algorithm and ignore workspace limit.
 *        - **None** (default): the behavior is determined by environment variable
 *        ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace
 *        (default), 2 for fastest.
 *
 *        - **workspace**: A large number leads to more (GPU) memory usage but may improve
 *        the performance.
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\convolution.cc:L154
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the ConvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel convolution kernel size: (h, w) or (d, h, w)
 * \param num_filter convolution filter(channel) number
 * \param stride convolution stride: (h, w) or (d, h, w)
 * \param dilate convolution dilate: (h, w) or (d, h, w)
 * \param pad pad for convolution: (h, w) or (d, h, w)
 * \param num_group Number of group partitions.
 * \param workspace Maximum temperal workspace allowed for convolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Convolution(const std::string& symbol_name,
                          Symbol data,
                          Symbol weight,
                          Symbol bias,
                          Shape kernel,
                          uint32_t num_filter,
                          Shape stride = Shape(),
                          Shape dilate = Shape(),
                          Shape pad = Shape(),
                          uint32_t num_group = 1,
                          uint64_t workspace = 1024,
                          bool no_bias = false,
                          ConvolutionCudnnTune cudnn_tune = ConvolutionCudnnTune::None,
                          bool cudnn_off = false,
                          ConvolutionLayout layout = ConvolutionLayout::None) {
  static const char *ConvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *ConvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NCW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", ConvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", ConvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*! \breif Whether to pick convolution algo by running performance test.
 *        Leads to higher startup time but may give faster speed. Options are:
 *        'off': no tuning
 *        'limited_workspace': run test and pick the fastest algorithm that doesn't
 *        'fastest': pick the fastest algorithm and ignore workspace limit.
 *        If set to None (default), behavior is determined by environment
 *        variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,
 *        1 for limited workspace (default), 2 for fastest.
 */
enum class Convolution_v1CudnnTune {
  None = 0,
  fastest = 1,
  limited_workspace = 2,
  off = 3
};

/*! \breif Set layout for input, output and weight. Empty for
 *        default layout: NCHW for 2d and NCDHW for 3d.
 */
enum class Convolution_v1Layout {
  None = 0,
  NCDHW = 1,
  NCHW = 2,
  NDHWC = 3,
  NHWC = 4
};

/*!
 * \breif This operator is DEPRECATED. Apply convolution to input then add a bias.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the ConvolutionV1Op.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel convolution kernel size: (h, w) or (d, h, w)
 * \param num_filter convolution filter(channel) number
 * \param stride convolution stride: (h, w) or (d, h, w)
 * \param dilate convolution dilate: (h, w) or (d, h, w)
 * \param pad pad for convolution: (h, w) or (d, h, w)
 * \param num_group Number of group partitions. Equivalent to slicing input into num_group
 *        partitions, apply convolution on each, then concatenate the results
 * \param workspace Maximum tmp workspace allowed for convolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 *        Leads to higher startup time but may give faster speed. Options are:
 *        'off': no tuning
 *        'limited_workspace': run test and pick the fastest algorithm that doesn't
 *        'fastest': pick the fastest algorithm and ignore workspace limit.
 *        If set to None (default), behavior is determined by environment
 *        variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,
 *        1 for limited workspace (default), 2 for fastest.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Convolution_v1(const std::string& symbol_name,
                             Symbol data,
                             Symbol weight,
                             Symbol bias,
                             Shape kernel,
                             uint32_t num_filter,
                             Shape stride = Shape(),
                             Shape dilate = Shape(),
                             Shape pad = Shape(),
                             uint32_t num_group = 1,
                             uint64_t workspace = 1024,
                             bool no_bias = false,
                             Convolution_v1CudnnTune cudnn_tune = Convolution_v1CudnnTune::None,
                             bool cudnn_off = false,
                             Convolution_v1Layout layout = Convolution_v1Layout::None) {
  static const char *Convolution_v1CudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *Convolution_v1LayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution_v1")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", Convolution_v1CudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", Convolution_v1LayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Applies correlation to inputs.
 * \param symbol_name name of the resulting symbol
 * \param data1 Input data1 to the correlation.
 * \param data2 Input data2 to the correlation.
 * \param kernel_size kernel size for Correlation must be an odd number
 * \param max_displacement Max displacement of Correlation
 * \param stride1 stride1 quantize data1 globally
 * \param stride2 stride2 quantize data2 within the neighborhood centered around data1
 * \param pad_size pad for Correlation
 * \param is_multiply operation type is either multiplication or subduction
 * \return new symbol
 */
inline Symbol Correlation(const std::string& symbol_name,
                          Symbol data1,
                          Symbol data2,
                          uint32_t kernel_size = 1,
                          uint32_t max_displacement = 1,
                          uint32_t stride1 = 1,
                          uint32_t stride2 = 1,
                          uint32_t pad_size = 0,
                          bool is_multiply = true) {
  return Operator("Correlation")
           .SetParam("kernel_size", kernel_size)
           .SetParam("max_displacement", max_displacement)
           .SetParam("stride1", stride1)
           .SetParam("stride2", stride2)
           .SetParam("pad_size", pad_size)
           .SetParam("is_multiply", is_multiply)
           .SetInput("data1", data1)
           .SetInput("data2", data2)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif
 *
 *        .. note:: `Crop` is deprecated. Use `slice` instead.
 *
 *        Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or
 *        with width and height of the second input symbol, i.e., with one input, we need
 *        specify the crop height and width, otherwise the second input symbol's size
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data Tensor or List of Tensors, the second input will be used as crop_like
 * \param num_args Number of inputs for crop, if equals one, then we will use the h_wfor
 *        crop height and width, else if equals two, then we will use the heightand width
 * \param offset crop offset coordinate: (y, x)
 * \param h_w crop height and width: (h, w)
 * \param center_crop If set to true, then it will use be the center_crop,or it will crop
 * \return new symbol
 */
inline Symbol Crop(const std::string& symbol_name,
                   const std::vector<Symbol>& data,
                   int num_args,
                   Shape offset = Shape(0,0),
                   Shape h_w = Shape(0,0),
                   bool center_crop = false) {
  return Operator("Crop")
           .SetParam("num_args", num_args)
           .SetParam("offset", offset)
           .SetParam("h_w", h_w)
           .SetParam("center_crop", center_crop)
(data)
           .CreateSymbol(symbol_name);
}

/*! \breif Whether to pick convolution algo by running performance test.
 */
enum class DeconvolutionCudnnTune {
  None = 0,
  fastest = 1,
  limited_workspace = 2,
  off = 3
};

/*! \breif Set layout for input, output and weight. Empty for
 *        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
 */
enum class DeconvolutionLayout {
  None = 0,
  NCDHW = 1,
  NCHW = 2,
  NCW = 3,
  NDHWC = 4,
  NHWC = 5
};

/*!
 * \breif Applies deconvolution to input and adds a bias.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the DeconvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel deconvolution kernel size: (h, w) or (d, h, w)
 * \param num_filter deconvolution filter(channel) number
 * \param stride deconvolution stride: (h, w) or (d, h, w)
 * \param dilate deconvolution dilate: (h, w) or (d, h, w)
 * \param pad pad for deconvolution: (h, w) or (d, h, w). A good number is :
 *        (kernel-1)/2. If target_shape is set, pad will be ignored and computed
 * \param adj adjustment for output shape: (h, w) or (d, h, w). If target_shape is set,
 * \param target_shape output shape with target shape : (h, w) or (d, h, w)
 * \param num_group number of groups partition
 * \param workspace Maximum temporal workspace allowed for deconvolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Deconvolution(const std::string& symbol_name,
                            Symbol data,
                            Symbol weight,
                            Symbol bias,
                            Shape kernel,
                            uint32_t num_filter,
                            Shape stride = Shape(),
                            Shape dilate = Shape(),
                            Shape pad = Shape(),
                            Shape adj = Shape(),
                            Shape target_shape = Shape(),
                            uint32_t num_group = 1,
                            uint64_t workspace = 512,
                            bool no_bias = true,
                            DeconvolutionCudnnTune cudnn_tune = DeconvolutionCudnnTune::None,
                            bool cudnn_off = false,
                            DeconvolutionLayout layout = DeconvolutionLayout::None) {
  static const char *DeconvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *DeconvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NCW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Deconvolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("adj", adj)
           .SetParam("target_shape", target_shape)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", DeconvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", DeconvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Applies dropout to input.
 *        During training, each element of the input is randomly set to zero with
 *        And then the whole tensor is rescaled by 1/(1-p) to keep the expectation the
 *        before applying dropout. During the test time, this behaves as an identity map.
 *
 * \param symbol_name name of the resulting symbol
 * \param data Input data to dropout.
 * \param p Fraction of the input that gets dropped out at training time
 * \return new symbol
 */
inline Symbol Dropout(const std::string& symbol_name,
                      Symbol data,
                      mx_float p = 0.5) {
  return Operator("Dropout")
           .SetParam("p", p)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Applies a linear transformation: :math:`Y = XW^T + b`.
 *
 *        Shapes:
 *
 *        - **data**: `(batch_size, input_dim)`
 *        - **weight**: `(num_hidden, input_dim)`
 *        - **bias**: `(num_hidden,)`
 *        - **out**: `(batch_size, num_hidden)`
 *
 *        The learnable parameters include both ``weight`` and ``bias``.
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\fully_connected.cc:L74
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param num_hidden Number of hidden nodes of the output.
 * \param no_bias Whether to disable bias parameter.
 * \return new symbol
 */
inline Symbol FullyConnected(const std::string& symbol_name,
                             Symbol data,
                             Symbol weight,
                             Symbol bias,
                             int num_hidden,
                             bool no_bias = false) {
  return Operator("FullyConnected")
           .SetParam("num_hidden", num_hidden)
           .SetParam("no_bias", no_bias)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*! \breif transformation type
 *        if transformation type is affine, data is affine matrix : (batch, 6)
 *        if transformation type is warp, data is optical flow : (batch, 2, h, w)
 */
enum class GridGeneratorTransformType {
  affine = 0,
  warp = 1
};

/*!
 * \breif Generates sampling grid for bilinear sampling.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the GridGeneratorOp.
 * \param transform_type transformation type
 *        if transformation type is affine, data is affine matrix : (batch, 6)
 *        if transformation type is warp, data is optical flow : (batch, 2, h, w)
 * \param target_shape if transformation type is affine, the operator need a target_shape
 *        if transofrmation type is warp, the operator will ignore target_shape
 * \return new symbol
 */
inline Symbol GridGenerator(const std::string& symbol_name,
                            Symbol data,
                            GridGeneratorTransformType transform_type,
                            Shape target_shape = Shape(0,0)) {
  static const char *GridGeneratorTransformTypeValues[] = {
    "affine",
    "warp"
  };
  return Operator("GridGenerator")
           .SetParam("transform_type", GridGeneratorTransformTypeValues[int(transform_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Applies instance normalization to the n-dimensional input array.
 *
 *        This operator takes an n-dimensional input array where (n>2) and normalizes
 *        the input using the following formula:
 *
 *        .. math::
 *
 *        out = \frac{x - mean[data]}{ \sqrt{Var[data]} + \epsilon} * gamma + beta
 *
 *        This layer is similar to batch normalization layer (`BatchNorm`)
 *        with two differences: first, the normalization is
 *        carried out per example (instance), not over a batch. Second, the
 *        same normalization is applied both at test and train time. This
 *        operation is also known as `contrast normalization`.
 *
 *        If the input data is of shape [batch, channel, spacial_dim1, spacial_dim2, ...],
 *        `gamma` and `beta` parameters must be vectors of shape [channel].
 *
 *        This implementation is based on paper:
 *
 *        .. [1] Instance Normalization: The Missing Ingredient for Fast Stylization,
 *        D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2).
 *
 *        Examples::
 *
 *        // Input of shape (2,1,2)
 *        x = [[[ 1.1,  2.2]],
 *        [[ 3.3,  4.4]]]
 *
 *        // gamma parameter of length 1
 *        gamma = [1.5]
 *
 *        // beta parameter of length 1
 *        beta = [0.5]
 *
 *        // Instance normalization is calculated with the above formula
 *        InstanceNorm(x,gamma,beta) = [[[-0.997527  ,  1.99752665]],
 *        [[-0.99752653,  1.99752724]]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\instance_norm.cc:L80
 * \param symbol_name name of the resulting symbol
 * \param data An n-dimensional input array (n > 2) of the form [batch, channel,
 * \param gamma A vector of length 'channel', which multiplies the normalized input.
 * \param beta A vector of length 'channel', which is added to the product of the
 * \param eps An `epsilon` parameter to prevent division by 0.
 * \return new symbol
 */
inline Symbol InstanceNorm(const std::string& symbol_name,
                           Symbol data,
                           Symbol gamma,
                           Symbol beta,
                           mx_float eps = 0.001) {
  return Operator("InstanceNorm")
           .SetParam("eps", eps)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol(symbol_name);
}

/*! \breif Specify the dimension along which to compute L2 norm.
 */
enum class L2NormalizationMode {
  channel = 0,
  instance = 1,
  spatial = 2
};

/*!
 * \breif Normalize the input array using the L2 norm.
 *
 *        For 1-D NDArray, it computes::
 *
 *        out = data / sqrt(sum(data ** 2) + eps)
 *
 *        For N-D NDArray, if the input array has shape (N, N, ..., N),
 *
 *        with ``mode`` = ``instance``, it normalizes each instance in the
 *        array by its L2 norm.::
 *
 *        for i in 0...N
 *        out[i,:,:,...,:] = data[i,:,:,...,:] / sqrt(sum(data[i,:,:,...,:] ** 2) + eps)
 *
 *        with ``mode`` = ``channel``, it normalizes each channel in the array by its L2
 *
 *        for i in 0...N
 *        out[:,i,:,...,:] = data[:,i,:,...,:] / sqrt(sum(data[:,i,:,...,:] ** 2) + eps)
 *
 *        with ``mode`` = ``spatial``, it normalizes the cross channel norm for each
 *        in the array by its L2 norm.::
 *
 *        for dim in 2...N
 *        for i in 0...N
 *        out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out,
 *        -dim-
 *
 *        Example::
 *
 *        x = [[[1,2],
 *        [3,4]],
 *        [[2,2],
 *        [5,6]]]
 *
 *        L2Normalization(x, mode='instance')
 *        =[[[ 0.18257418  0.36514837]
 *        [ 0.54772252  0.73029673]]
 *        [[ 0.24077171  0.24077171]
 *        [ 0.60192931  0.72231513]]]
 *
 *        L2Normalization(x, mode='channel')
 *        =[[[ 0.31622776  0.44721359]
 *        [ 0.94868326  0.89442718]]
 *        [[ 0.37139067  0.31622776]
 *        [ 0.92847669  0.94868326]]]
 *
 *        L2Normalization(x, mode='spatial')
 *        =[[[ 0.44721359  0.89442718]
 *        [ 0.60000002  0.80000001]]
 *        [[ 0.70710677  0.70710677]
 *        [ 0.6401844   0.76822126]]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\l2_normalization.cc:L74
 * \param symbol_name name of the resulting symbol
 * \param data Input array to normalize.
 * \param eps A small constant for numerical stability.
 * \param mode Specify the dimension along which to compute L2 norm.
 * \return new symbol
 */
inline Symbol L2Normalization(const std::string& symbol_name,
                              Symbol data,
                              mx_float eps = 1e-10,
                              L2NormalizationMode mode = L2NormalizationMode::instance) {
  static const char *L2NormalizationModeValues[] = {
    "channel",
    "instance",
    "spatial"
  };
  return Operator("L2Normalization")
           .SetParam("eps", eps)
           .SetParam("mode", L2NormalizationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Applies local response normalization to the input.
 *
 *        The local response normalization layer performs "lateral inhibition" by
 *        over local input regions.
 *
 *        If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel
 *        :math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized
 *        activity :math:`b_{x,y}^{i}` is given by the expression:
 *
 *        .. math::
 *        b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \alpha \sum_{j=max(0,
 *
 *        where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial
 *        number of kernels in the layer.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \param nsize normalization window width in elements.
 * \param alpha The variance scaling parameter :math:`lpha` in the LRN expression.
 * \param beta The power parameter :math:`eta` in the LRN expression.
 * \param knorm The parameter :math:`k` in the LRN expression.
 * \return new symbol
 */
inline Symbol LRN(const std::string& symbol_name,
                  Symbol data,
                  uint32_t nsize,
                  mx_float alpha = 0.0001,
                  mx_float beta = 0.75,
                  mx_float knorm = 2) {
  return Operator("LRN")
           .SetParam("nsize", nsize)
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetParam("knorm", knorm)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif If this is set to null, the output gradient will not be normalized. If this is
 *        set to batch, the output gradient will be divided by the batch size. If this is
 *        set to valid, the output gradient will be divided by the number of valid input
 */
enum class MakeLossNormalization {
  batch = 0,
  null = 1,
  valid = 2
};

/*!
 * \breif Make your own loss function in network construction.
 *
 *        This operator accepts a customized loss function symbol as a terminal loss and
 *        the symbol should be an operator with no backward dependency.
 *        The output of this function is the gradient of loss with respect to the input
 *
 *        For example, if you are a making a cross entropy loss function. Assume ``out``
 *        predicted output and ``label`` is the true label, then the cross entropy can be
 *
 *        cross_entropy = label * log(out) + (1 - label) * log(1 - out)
 *        loss = MakeLoss(cross_entropy)
 *
 *        We will need to use ``MakeLoss`` when we are creating our own loss function or
 *        combine multiple loss functions. Also we may want to stop some variables'
 *        from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.
 *
 *        In addition, we can give a scale to the loss by setting ``grad_scale``,
 *        so that the gradient of the loss will be rescaled in the backpropagation.
 *
 *        .. note:: This operator should be used as a Symbol instead of NDArray.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \param grad_scale Gradient scale as a supplement to unary and binary operators
 * \param valid_thresh clip each element in the array to 0 when it is less than
 * \param normalization If this is set to null, the output gradient will not be
 *        normalized. If this is set to batch, the output gradient will be divided by the
 *        batch size. If this is set to valid, the output gradient will be divided by the
 * \return new symbol
 */
inline Symbol MakeLoss(const std::string& symbol_name,
                       Symbol data,
                       mx_float grad_scale = 1,
                       mx_float valid_thresh = 0,
                       MakeLossNormalization normalization = MakeLossNormalization::null) {
  static const char *MakeLossNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("MakeLoss")
           .SetParam("grad_scale", grad_scale)
           .SetParam("valid_thresh", valid_thresh)
           .SetParam("normalization", MakeLossNormalizationValues[int(normalization)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Pooling type to be applied.
 */
enum class PoolingPoolType {
  avg = 0,
  max = 1,
  sum = 2
};

/*! \breif Pooling convention to be applied.
 */
enum class PoolingPoolingConvention {
  full = 0,
  valid = 1
};

/*!
 * \breif Performs pooling on the input.
 *
 *        The shapes for 1-D pooling are
 *
 *        - **data**: *(batch_size, channel, width)*,
 *        - **out**: *(batch_size, num_filter, out_width)*.
 *
 *        The shapes for 2-D pooling are
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*, with::
 *
 *        out_height = f(height, kernel[0], pad[0], stride[0])
 *        out_width = f(width, kernel[1], pad[1], stride[1])
 *
 *        The defintion of *f* depends on ``pooling_convention``, which has two options:
 *
 *        - **valid** (default)::
 *
 *        f(x, k, p, s) = floor((x+2*p-k)/s)+1
 *
 *        - **full**, which is compatible with Caffe::
 *
 *        f(x, k, p, s) = ceil((x+2*p-k)/s)+1
 *
 *        But ``global_pool`` is set to be true, then do a global pooling, namely reset
 *        ``kernel=(height, width)``.
 *
 *        Three pooling options are supported by ``pool_type``:
 *
 *        - **avg**: average pooling
 *        - **max**: max pooling
 *        - **sum**: sum pooling
 *
 *        For 3-D pooling, an additional *depth* dimension is added before
 *        *height*. Namely the input data will have shape *(batch_size, channel, depth,
 *        height, width)*.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the pooling operator.
 * \param kernel pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current input
 * \param cudnn_off Turn off cudnn pooling and use MXNet pooling operator.
 * \param pooling_convention Pooling convention to be applied.
 * \param stride stride: for pooling (y, x) or (d, y, x)
 * \param pad pad for pooling: (y, x) or (d, y, x)
 * \return new symbol
 */
inline Symbol Pooling(const std::string& symbol_name,
                      Symbol data,
                      Shape kernel,
                      PoolingPoolType pool_type,
                      bool global_pool = false,
                      bool cudnn_off = false,
                      PoolingPoolingConvention pooling_convention = PoolingPoolingConvention::valid,
                      Shape stride = Shape(),
                      Shape pad = Shape()) {
  static const char *PoolingPoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  static const char *PoolingPoolingConventionValues[] = {
    "full",
    "valid"
  };
  return Operator("Pooling")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", PoolingPoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("pooling_convention", PoolingPoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Pooling type to be applied.
 */
enum class Pooling_v1PoolType {
  avg = 0,
  max = 1,
  sum = 2
};

/*! \breif Pooling convention to be applied.
 */
enum class Pooling_v1PoolingConvention {
  full = 0,
  valid = 1
};

/*!
 * \breif This operator is DEPRECATED.
 *        Perform pooling on the input.
 *
 *        The shapes for 2-D pooling is
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*, with::
 *
 *        out_height = f(height, kernel[0], pad[0], stride[0])
 *        out_width = f(width, kernel[1], pad[1], stride[1])
 *
 *        The defintion of *f* depends on ``pooling_convention``, which has two options:
 *
 *        - **valid** (default)::
 *
 *        f(x, k, p, s) = floor((x+2*p-k)/s)+1
 *
 *        - **full**, which is compatible with Caffe::
 *
 *        f(x, k, p, s) = ceil((x+2*p-k)/s)+1
 *
 *        But ``global_pool`` is set to be true, then do a global pooling, namely reset
 *        ``kernel=(height, width)``.
 *
 *        Three pooling options are supported by ``pool_type``:
 *
 *        - **avg**: average pooling
 *        - **max**: max pooling
 *        - **sum**: sum pooling
 *
 *        1-D pooling is special case of 2-D pooling with *weight=1* and
 *        *kernel[1]=1*.
 *
 *        For 3-D pooling, an additional *depth* dimension is added before
 *        *height*. Namely the input data will have shape *(batch_size, channel, depth,
 *        height, width)*.
 *
 *
 *
 *        Defined in
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the pooling operator.
 * \param kernel pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current input
 * \param pooling_convention Pooling convention to be applied.
 * \param stride stride: for pooling (y, x) or (d, y, x)
 * \param pad pad for pooling: (y, x) or (d, y, x)
 * \return new symbol
 */
inline Symbol Pooling_v1(const std::string& symbol_name,
                         Symbol data,
                         Shape kernel,
                         Pooling_v1PoolType pool_type,
                         bool global_pool = false,
                         Pooling_v1PoolingConvention pooling_convention = Pooling_v1PoolingConvention::valid,
                         Shape stride = Shape(),
                         Shape pad = Shape()) {
  static const char *Pooling_v1PoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  static const char *Pooling_v1PoolingConventionValues[] = {
    "full",
    "valid"
  };
  return Operator("Pooling_v1")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", Pooling_v1PoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("pooling_convention", Pooling_v1PoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes and optimizes for squared loss.
 *
 *        .. note::
 *        Use the LinearRegressionOutput as the final output layer of a net.
 *
 *        By default, gradients of this loss function are scaled by factor `1/n`, where n
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\regression_output.cc:L45
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LinearRegressionOutput(const std::string& symbol_name,
                                     Symbol data,
                                     Symbol label,
                                     mx_float grad_scale = 1) {
  return Operator("LinearRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes mean absolute error of the input.
 *
 *        MAE is a risk metric corresponding to the expected value of the absolute error.
 *
 *        If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i`
 *        then the mean absolute error (MAE) estimated over :math:`n` samples is defined
 *
 *        :math:`\text{MAE}(y, \hat{y} ) = \frac{1}{n} \sum_{i=0}^{n-1} \left| y_i -
 *
 *        .. note::
 *        Use the MAERegressionOutput as the final output layer of a net.
 *
 *        By default, gradients of this loss function are scaled by factor `1/n`, where n
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\regression_output.cc:L66
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol MAERegressionOutput(const std::string& symbol_name,
                                  Symbol data,
                                  Symbol label,
                                  mx_float grad_scale = 1) {
  return Operator("MAERegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Applies a logistic function to the input.
 *
 *        The logistic function, also known as the sigmoid function, is computed as
 *        :math:`\frac{1}{1+exp(-x)}`.
 *
 *        Commonly, the sigmoid is used to squash the real-valued output of a linear model
 *        :math:wTx+b into the [0,1] range so that it can be interpreted as a probability.
 *        It is suitable for binary classification or probability prediction tasks.
 *
 *        .. note::
 *        Use the LogisticRegressionOutput as the final output layer of a net.
 *
 *        By default, gradients of this loss function are scaled by factor `1/n`, where n
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\regression_output.cc:L87
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LogisticRegressionOutput(const std::string& symbol_name,
                                       Symbol data,
                                       Symbol label,
                                       mx_float grad_scale = 1) {
  return Operator("LogisticRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*! \breif the type of RNN to compute
 */
enum class RNNMode {
  gru = 0,
  lstm = 1,
  rnn_relu = 2,
  rnn_tanh = 3
};

/*!
 * \breif Applies a recurrent layer to input.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to RNN
 * \param parameters Vector of all RNN trainable parameters concatenated
 * \param state initial hidden state of the RNN
 * \param state_cell initial cell state for LSTM networks (only for LSTM)
 * \param state_size size of the state for each layer
 * \param num_layers number of stacked layers
 * \param mode the type of RNN to compute
 * \param bidirectional whether to use bidirectional recurrent layers
 * \param p Dropout probability, fraction of the input that gets dropped out at training
 * \param state_outputs Whether to have the states as symbol outputs.
 * \return new symbol
 */
inline Symbol RNN(const std::string& symbol_name,
                  Symbol data,
                  Symbol parameters,
                  Symbol state,
                  Symbol state_cell,
                  uint32_t state_size,
                  uint32_t num_layers,
                  RNNMode mode,
                  bool bidirectional = false,
                  mx_float p = 0,
                  bool state_outputs = false) {
  static const char *RNNModeValues[] = {
    "gru",
    "lstm",
    "rnn_relu",
    "rnn_tanh"
  };
  return Operator("RNN")
           .SetParam("state_size", state_size)
           .SetParam("num_layers", num_layers)
           .SetParam("mode", RNNModeValues[int(mode)])
           .SetParam("bidirectional", bidirectional)
           .SetParam("p", p)
           .SetParam("state_outputs", state_outputs)
           .SetInput("data", data)
           .SetInput("parameters", parameters)
           .SetInput("state", state)
           .SetInput("state_cell", state_cell)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Performs region of interest(ROI) pooling on the input array.
 *
 *        ROI pooling is a variant of a max pooling layer, in which the output size is
 *        region of interest is a parameter. Its purpose is to perform max pooling on the
 *        of non-uniform sizes to obtain fixed-size feature maps. ROI pooling is a
 *        layer mostly used in training a `Fast R-CNN` network for object detection.
 *
 *        This operator takes a 4D feature map as an input array and region proposals as
 *        then it pools over sub-regions of input and produces a fixed-sized output array
 *        regardless of the ROI size.
 *
 *        To crop the feature map accordingly, you can resize the bounding box coordinates
 *        by changing the parameters `rois` and `spatial_scale`.
 *
 *        The cropped feature maps are pooled by standard max pooling operation to a
 *        indicated by a `pooled_size` parameter. batch_size will change to the number of
 *        bounding boxes after `ROIPooling`.
 *
 *        The size of each region of interest doesn't have to be perfectly divisible by
 *        the number of pooling sections(`pooled_size`).
 *
 *        Example::
 *
 *        x = [[[[  0.,   1.,   2.,   3.,   4.,   5.],
 *        [  6.,   7.,   8.,   9.,  10.,  11.],
 *        [ 12.,  13.,  14.,  15.,  16.,  17.],
 *        [ 18.,  19.,  20.,  21.,  22.,  23.],
 *        [ 24.,  25.,  26.,  27.,  28.,  29.],
 *        [ 30.,  31.,  32.,  33.,  34.,  35.],
 *        [ 36.,  37.,  38.,  39.,  40.,  41.],
 *        [ 42.,  43.,  44.,  45.,  46.,  47.]]]]
 *
 *        // region of interest i.e. bounding box coordinates.
 *        y = [[0,0,0,4,4]]
 *
 *        // returns array of shape (2,2) according to the given roi with max pooling.
 *        ROIPooling(x, y, (2,2), 1.0) = [[[[ 14.,  16.],
 *        [ 26.,  28.]]]]
 *
 *        // region of interest is changed due to the change in `spacial_scale` parameter.
 *        ROIPooling(x, y, (2,2), 0.7) = [[[[  7.,   9.],
 *        [ 19.,  21.]]]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\roi_pooling.cc:L273
 * \param symbol_name name of the resulting symbol
 * \param data The input array to the pooling operator,  a 4D Feature maps
 * \param rois Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]],
 *        where (x1, y1) and (x2, y2) are top left and bottom right corners of designated
 *        region of interest. `batch_index` indicates the index of corresponding image in
 * \param pooled_size ROI pooling output shape (h,w)
 * \param spatial_scale Ratio of input feature map height (or w) to raw image height (or
 * \return new symbol
 */
inline Symbol ROIPooling(const std::string& symbol_name,
                         Symbol data,
                         Symbol rois,
                         Shape pooled_size,
                         mx_float spatial_scale) {
  return Operator("ROIPooling")
           .SetParam("pooled_size", pooled_size)
           .SetParam("spatial_scale", spatial_scale)
           .SetInput("data", data)
           .SetInput("rois", rois)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Takes the last element of a sequence.
 *
 *        This function takes an n-dimensional input array of the form
 *        [max_sequence_length, batch_size, other_feature_dims] and returns a
 *        of the form [batch_size, other_feature_dims].
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        an input array of positive ints of dimension [batch_size]. To use this
 *        set `use_sequence_length` to `True`, otherwise each example in the batch is
 *        to have the max sequence length.
 *
 *        .. note:: Alternatively, you can also use `take` operator.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.],
 *        [  7.,   8.,   9.]],
 *
 *        [[ 10.,   11.,   12.],
 *        [ 13.,   14.,   15.],
 *        [ 16.,   17.,   18.]],
 *
 *        [[  19.,   20.,   21.],
 *        [  22.,   23.,   24.],
 *        [  25.,   26.,   27.]]]
 *
 *        // returns last sequence when sequence_length parameter is not used
 *        SequenceLast(x) = [[  19.,   20.,   21.],
 *        [  22.,   23.,   24.],
 *        [  25.,   26.,   27.]]
 *
 *        // sequence_length y is used
 *        SequenceLast(x, y=[1,1,1], use_sequence_length=True) =
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.],
 *        [  7.,   8.,   9.]]
 *
 *        // sequence_length y is used
 *        SequenceLast(x, y=[1,2,3], use_sequence_length=True) =
 *        [[  1.,    2.,   3.],
 *        [  13.,  14.,  15.],
 *        [  25.,  26.,  27.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\sequence_last.cc:L77
 * \param symbol_name name of the resulting symbol
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \return new symbol
 */
inline Symbol SequenceLast(const std::string& symbol_name,
                           Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false) {
  return Operator("SequenceLast")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Sets all elements outside the sequence to a constant value.
 *
 *        This function takes an n-dimensional input array of the form
 *        [max_sequence_length, batch_size, other_feature_dims] and returns an array of
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        should be an input array of positive ints of dimension [batch_size].
 *        To use this parameter, set `use_sequence_length` to `True`,
 *        otherwise each example in the batch is assumed to have the max sequence length
 *        this operator works as the `identity` operator.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // Batch 1
 *        B1 = [[  1.,   2.,   3.],
 *        [  7.,   8.,   9.],
 *        [ 13.,  14.,  15.]]
 *
 *        // Batch 2
 *        B2 = [[  4.,   5.,   6.],
 *        [ 10.,  11.,  12.],
 *        [ 16.,  17.,  18.]]
 *
 *        // works as identity operator when sequence_length parameter is not used
 *        SequenceMask(x) = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // sequence_length [1,1] means 1 of each batch will be kept
 *        // and other rows are masked with default mask value = 0
 *        SequenceMask(x, y=[1,1], use_sequence_length=True) =
 *        [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  0.,   0.,   0.],
 *        [  0.,   0.,   0.]],
 *
 *        [[  0.,   0.,   0.],
 *        [  0.,   0.,   0.]]]
 *
 *        // sequence_length [2,3] means 2 of batch B1 and 3 of batch B2 will be kept
 *        // and other rows are masked with value = 1
 *        SequenceMask(x, y=[2,3], use_sequence_length=True, value=1) =
 *        [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [  10.,  11.,  12.]],
 *
 *        [[   1.,   1.,   1.],
 *        [  16.,  17.,  18.]]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\sequence_mask.cc:L112
 * \param symbol_name name of the resulting symbol
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \param value The value to be used as a mask.
 * \return new symbol
 */
inline Symbol SequenceMask(const std::string& symbol_name,
                           Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false,
                           mx_float value = 0) {
  return Operator("SequenceMask")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetParam("value", value)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Reverses the elements of each sequence.
 *
 *        This function takes an n-dimensional input array of the form
 *        and returns an array of the same shape.
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        `sequence_length` should be an input array of positive ints of dimension
 *        To use this parameter, set `use_sequence_length` to `True`,
 *        otherwise each example in the batch is assumed to have the max sequence length.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // Batch 1
 *        B1 = [[  1.,   2.,   3.],
 *        [  7.,   8.,   9.],
 *        [ 13.,  14.,  15.]]
 *
 *        // Batch 2
 *        B2 = [[  4.,   5.,   6.],
 *        [ 10.,  11.,  12.],
 *        [ 16.,  17.,  18.]]
 *
 *        // returns reverse sequence when sequence_length parameter is not used
 *        SequenceReverse(x) = [[[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]]]
 *
 *        // sequence_length [2,2] means 2 rows of
 *        // both batch B1 and B2 will be reversed.
 *        SequenceReverse(x, y=[2,2], use_sequence_length=True) =
 *        [[[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // sequence_length [2,3] means 2 of batch B2 and 3 of batch B3
 *        // will be reversed.
 *        SequenceReverse(x, y=[2,3], use_sequence_length=True) =
 *        [[[  7.,   8.,   9.],
 *        [ 16.,  17.,  18.]],
 *
 *        [[  1.,   2.,   3.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14,   15.],
 *        [  4.,   5.,   6.]]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\sequence_reverse.cc:L98
 * \param symbol_name name of the resulting symbol
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \return new symbol
 */
inline Symbol SequenceReverse(const std::string& symbol_name,
                              Symbol data,
                              Symbol sequence_length,
                              bool use_sequence_length = false) {
  return Operator("SequenceReverse")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*! \breif Softmax Mode. If set to instance, this operator will compute a softmax for each
 *        instance in the batch; this is the default mode. If set to channel, this
 *        operator will compute a num_channel-class softmax at each position of each
 *        instance; this can be used for fully convolutional network, image segmentation,
 */
enum class SoftmaxActivationMode {
  channel = 0,
  instance = 1
};

/*!
 * \breif Applies softmax activation to input. This is intended for internal layers. For
 *        output (loss layer) please use SoftmaxOutput. If mode=instance, this operator
 *        will compute a softmax for each instance in the batch; this is the default
 *        mode. If mode=channel, this operator will compute a num_channel-class softmax
 *        at each position of each instance; this can be used for fully convolutional
 * \param symbol_name name of the resulting symbol
 * \param data Input data to activation function.
 * \param mode Softmax Mode. If set to instance, this operator will compute a softmax for
 *        each instance in the batch; this is the default mode. If set to channel, this
 *        operator will compute a num_channel-class softmax at each position of each
 *        instance; this can be used for fully convolutional network, image segmentation,
 * \return new symbol
 */
inline Symbol SoftmaxActivation(const std::string& symbol_name,
                                Symbol data,
                                SoftmaxActivationMode mode = SoftmaxActivationMode::instance) {
  static const char *SoftmaxActivationModeValues[] = {
    "channel",
    "instance"
  };
  return Operator("SoftmaxActivation")
           .SetParam("mode", SoftmaxActivationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif Normalize the gradient
 */
enum class SoftmaxOutputNormalization {
  batch = 0,
  null = 1,
  valid = 2
};

/*!
 * \breif Computes softmax with logit loss.
 *
 *        In the forward pass, the softmax output is returned. Assume the input data has
 *        shape *(n,k)*, then the output will have the same shape as the input, which is
 *
 *        .. math::
 *        out[i,:] = softmax(data[i,:])
 *
 *        for :math:`i=0,...,n-1`, where
 *
 *        .. math::
 *        softmax(x) = \left[..., \frac{exp(x[j])}{exp(x[0])+...+exp(x[k-1])}, ...\right]
 *
 *        For general *N*-D input array with shape :math:`(d_1, ..., d_n)`. Denoted by
 *        :math:`s=d_1d_2...d_n`. The way to compute softmax various:
 *
 *        - ``preserve_shape`` is false (default). Reshape input into a 2-D array with
 *        shape :math:`(d_1, s/d_1)` beforing computing the softmax, and then reshaped
 *        original shape.
 *
 *        - ``preserve_shape`` is true. For all :math:`i_1, ..., i_{n-1}`, compute
 *
 *        .. math::
 *        out[i_1, ..., i_{n-1}, :] = softmax(data[i_1, ..., i_{n-1},:])
 *
 *        - ``multi_output`` is true. For all :math:`i_1, ..., i_{n-1}`, compute
 *
 *        .. math::
 *        out[i_1, :, ..., i_{n-1}] = softmax(data[i_1, :, ..., i_{n-1}])
 *
 *        In the backward pass, the logit loss, also called cross-entroy loss, is
 *        added. The provided label can be a *(N-1)*-D label index array or a *N*-D label
 *        probability array.
 *
 *        Examples with a particular label can be ignored during backward by specifying
 *        ``ignore_label`` (also need ``use_ignore`` to be true).
 *
 *        A scale can be applied to the gradient by ``grad_scale``, which is often used in
 *        mutli-loss object function in which we can given each loss different weight. It
 *        also supports various ways to normalize the gradient by ``normalization``:
 *
 *        - **null**: do nothing
 *        - **batch**: divide by batch size (number of examples)
 *        - **valid**: divide by the number of examples which are not ignored.
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\softmax_output.cc:L77
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \param label Ground truth label.
 * \param grad_scale Scale the gradient by a float factor
 * \param ignore_label the labels with value equals to ``ignore_label`` will be ignored
 * \param multi_output If set to true, softmax will applied on axis 1
 * \param use_ignore If set to true, the ignore_label value will not contribute to the
 * \param preserve_shape If true, softmax will applied on the last axis
 * \param normalization Normalize the gradient
 * \param out_grad Apply weighting from output gradient
 * \return new symbol
 */
inline Symbol SoftmaxOutput(const std::string& symbol_name,
                            Symbol data,
                            Symbol label,
                            mx_float grad_scale = 1,
                            mx_float ignore_label = -1,
                            bool multi_output = false,
                            bool use_ignore = false,
                            bool preserve_shape = false,
                            SoftmaxOutputNormalization normalization = SoftmaxOutputNormalization::null,
                            bool out_grad = false) {
  static const char *SoftmaxOutputNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("SoftmaxOutput")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxOutputNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*! \breif Normalize the gradient
 */
enum class SoftmaxNormalization {
  batch = 0,
  null = 1,
  valid = 2
};

/*!
 * \breif Perform a softmax transformation on input. Please use SoftmaxOutput.. note::
 * \param symbol_name name of the resulting symbol
 * \param data Input data to softmax.
 * \param grad_scale Scale the gradient by a float factor
 * \param ignore_label the labels with value equals to ``ignore_label`` will be ignored
 * \param multi_output If set to true, softmax will applied on axis 1
 * \param use_ignore If set to true, the ignore_label value will not contribute to the
 * \param preserve_shape If true, softmax will applied on the last axis
 * \param normalization Normalize the gradient
 * \param out_grad Apply weighting from output gradient
 * \return new symbol
 */
inline Symbol Softmax(const std::string& symbol_name,
                      Symbol data,
                      mx_float grad_scale = 1,
                      mx_float ignore_label = -1,
                      bool multi_output = false,
                      bool use_ignore = false,
                      bool preserve_shape = false,
                      SoftmaxNormalization normalization = SoftmaxNormalization::null,
                      bool out_grad = false) {
  static const char *SoftmaxNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("Softmax")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \breif transformation type
 */
enum class SpatialTransformerTransformType {
  affine = 0
};

/*! \breif sampling type
 */
enum class SpatialTransformerSamplerType {
  bilinear = 0
};

/*!
 * \breif Applies a spatial transformer to input feature map.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the SpatialTransformerOp.
 * \param loc localisation net, the output dim should be 6 when transform_type is affine.
 * \param transform_type transformation type
 * \param sampler_type sampling type
 * \param target_shape output shape(h, w) of spatial transformer: (y, x)
 * \return new symbol
 */
inline Symbol SpatialTransformer(const std::string& symbol_name,
                                 Symbol data,
                                 Symbol loc,
                                 SpatialTransformerTransformType transform_type,
                                 SpatialTransformerSamplerType sampler_type,
                                 Shape target_shape = Shape(0,0)) {
  static const char *SpatialTransformerTransformTypeValues[] = {
    "affine"
  };
  static const char *SpatialTransformerSamplerTypeValues[] = {
    "bilinear"
  };
  return Operator("SpatialTransformer")
           .SetParam("transform_type", SpatialTransformerTransformTypeValues[int(transform_type)])
           .SetParam("sampler_type", SpatialTransformerSamplerTypeValues[int(sampler_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .SetInput("loc", loc)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Computes support vector machine based transformation of the input.
 *
 *        This tutorial demonstrates using SVM as output layer for classification instead
 *        https://github.com/dmlc/mxnet/tree/master/example/svm_mnist.
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data Input data for SVM transformation.
 * \param label Class label for the input data.
 * \param margin The loss function penalizes outputs that lie outside this margin.
 * \param regularization_coefficient Regularization parameter for the SVM. This balances
 * \param use_linear Whether to use L1-SVM objective. L2-SVM objective is used by default.
 * \return new symbol
 */
inline Symbol SVMOutput(const std::string& symbol_name,
                        Symbol data,
                        Symbol label,
                        mx_float margin = 1,
                        mx_float regularization_coefficient = 1,
                        bool use_linear = false) {
  return Operator("SVMOutput")
           .SetParam("margin", margin)
           .SetParam("regularization_coefficient", regularization_coefficient)
           .SetParam("use_linear", use_linear)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Apply a custom operator implemented in a frontend language (like Python).
 *
 *        Custom operators should override required methods like `forward` and `backward`.
 *        The custom operator must be registered before it can be used.
 *        Please check the tutorial here: http://mxnet.io/how_to/new_op.html.
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param op_type Name of the custom operator. This is the name that is passed to
 * \param data Input data for the custom operator.
 * \return new symbol
 */
inline Symbol Custom(const std::string& symbol_name,
                     const std::string& op_type,
                     Symbol data) {
  return Operator("Custom")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \breif Calculate cross_entropy(data, one_hot(label))
 *
 *        From:E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\loss_binary_op.cc:12
 * \param data Input data
 * \param label Input label
 * \return new symbol
 */
inline Symbol softmax_cross_entropy(Symbol data,
                                    Symbol label) {
  return Operator("softmax_cross_entropy")
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Update function for Stochastic Gradient Descent (SDG) optimizer.
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * gradient
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\optimizer_op.cc:L25
 * \param weight Weight
 * \param grad Gradient
 * \param lr Learning rate
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol sgd_update(Symbol weight,
                         Symbol grad,
                         mx_float lr,
                         mx_float wd = 0,
                         mx_float rescale_grad = 1,
                         mx_float clip_gradient = -1) {
  return Operator("sgd_update")
           .SetParam("lr", lr)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .CreateSymbol();
}

/*!
 * \breif Momentum update function for Stochastic Gradient Descent (SDG) optimizer.
 *
 *        Momentum update has better convergence rates on neural networks. Mathematically
 *        like below:
 *
 *        .. math::
 *
 *        v_1 = \alpha * \nabla J(W_0)\\
 *        v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} + v_t
 *
 *        It updates the weights using::
 *
 *        v = momentum * v - learning_rate * gradient
 *        weight += v
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\optimizer_op.cc:L55
 * \param weight Weight
 * \param grad Gradient
 * \param mom Momentum
 * \param lr Learning rate
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol sgd_mom_update(Symbol weight,
                             Symbol grad,
                             Symbol mom,
                             mx_float lr,
                             mx_float momentum = 0,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1) {
  return Operator("sgd_mom_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mom", mom)
           .CreateSymbol();
}

/*!
 * \breif Update function for Adam optimizer. Adam is seen as a generalization
 *        of AdaGrad.
 *
 *        Adam update consists of the following steps, where g represents gradient and m,
 *        are 1st and 2nd order moment estimates (mean and variance).
 *
 *        .. math::
 *
 *        g_t = \nabla J(W_{t-1})\\
 *        m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
 *        v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
 *        W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }
 *
 *        It updates the weights using::
 *
 *        m = beta1*m + (1-beta1)*grad
 *        v = beta2*v + (1-beta2)*(grad**2)
 *        w += - learning_rate * m / (sqrt(v) + epsilon)
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\optimizer_op.cc:L91
 * \param weight Weight
 * \param grad Gradient
 * \param mean Moving mean
 * \param var Moving variance
 * \param lr Learning rate
 * \param beta1 The decay rate for the 1st moment estimates.
 * \param beta2 The decay rate for the 2nd moment estimates.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol adam_update(Symbol weight,
                          Symbol grad,
                          Symbol mean,
                          Symbol var,
                          mx_float lr,
                          mx_float beta1 = 0.9,
                          mx_float beta2 = 0.999,
                          mx_float epsilon = 1e-08,
                          mx_float wd = 0,
                          mx_float rescale_grad = 1,
                          mx_float clip_gradient = -1) {
  return Operator("adam_update")
           .SetParam("lr", lr)
           .SetParam("beta1", beta1)
           .SetParam("beta2", beta2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mean", mean)
           .SetInput("var", var)
           .CreateSymbol();
}

/*!
 * \breif Update function for RMSProp optimizer. The RMSProp code follows the version in
 *        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\optimizer_op.cc:L111
 * \param weight Weight
 * \param grad Gradient
 * \param n n
 * \param lr Learning rate
 * \param gamma1 The dacay rate of momentum estimates.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param clip_weights Clip weights to the range of [-clip_weights, clip_weights] If
 *        clip_weights <= 0, weight clipping is turned off. weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmsprop_update(Symbol weight,
                             Symbol grad,
                             Symbol n,
                             mx_float lr,
                             mx_float gamma1 = 0.95,
                             mx_float epsilon = 1e-08,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1,
                             mx_float clip_weights = -1) {
  return Operator("rmsprop_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("n", n)
           .CreateSymbol();
}

/*!
 * \breif Update function for RMSPropAlex optimizer. The RMSPropAlex code follows the
 *        http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\optimizer_op.cc:L130
 * \param weight Weight
 * \param grad Gradient
 * \param n n
 * \param g g
 * \param delta delta
 * \param lr Learning rate
 * \param gamma1 Decay rate.
 * \param gamma2 Decay rate.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param clip_weights Clip weights to the range of [-clip_weights, clip_weights] If
 *        clip_weights <= 0, weight clipping is turned off. weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmspropalex_update(Symbol weight,
                                 Symbol grad,
                                 Symbol n,
                                 Symbol g,
                                 Symbol delta,
                                 mx_float lr,
                                 mx_float gamma1 = 0.95,
                                 mx_float gamma2 = 0.9,
                                 mx_float epsilon = 1e-08,
                                 mx_float wd = 0,
                                 mx_float rescale_grad = 1,
                                 mx_float clip_gradient = -1,
                                 mx_float clip_weights = -1) {
  return Operator("rmspropalex_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("gamma2", gamma2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("n", n)
           .SetInput("g", g)
           .SetInput("delta", delta)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise sum of the input arrays with broadcasting.
 *
 *        `broadcast_plus` is an alias to the function `broadcast_add`.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_add(x, y) = [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]
 *
 *        broadcast_plus(x, y) = [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L32
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_add(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise difference of the input arrays with broadcasting.
 *
 *        `broadcast_minus` is an alias to the function `broadcast_sub`.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_sub(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *        broadcast_minus(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L71
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_sub(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_sub")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise product of the input arrays with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_mul(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L104
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_mul(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_mul")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise division of the input arrays with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 6.,  6.,  6.],
 *        [ 6.,  6.,  6.]]
 *
 *        y = [[ 2.],
 *        [ 3.]]
 *
 *        broadcast_div(x, y) = [[ 3.,  3.,  3.],
 *        [ 2.,  2.,  2.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L137
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_div(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_div")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Adds arguments element-wise.
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_add(Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise squared value of the input.
 *
 *        .. math::
 *        square(x) = x^2
 *
 *        Example::
 *
 *        square([2, 3, 4]) = [3, 9, 16]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L269
 * \param data The input array.
 * \return new symbol
 */
inline Symbol square(Symbol data) {
  return Operator("square")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Converts each element of the input array from degrees to radians.
 *
 *        .. math::
 *        radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L507
 * \param data The input array.
 * \return new symbol
 */
inline Symbol radians(Symbol data) {
  return Operator("radians")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise square-root value of the input.
 *
 *        .. math::
 *        \textrm{sqrt}(x) = \sqrt{x}
 *
 *        Example::
 *
 *        sqrt([4, 9, 16]) = [2, 3, 4]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L287
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sqrt(Symbol data) {
  return Operator("sqrt")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the hyperbolic cosine  of the input array, computed element-wise.
 *
 *        .. math::
 *        cosh(x) = 0.5\times(exp(x) + exp(-x))
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L535
 * \param data The input array.
 * \return new symbol
 */
inline Symbol cosh(Symbol data) {
  return Operator("cosh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise inverse square-root value of the input.
 *
 *        .. math::
 *        rsqrt(x) = 1/\sqrt{x}
 *
 *        Example::
 *
 *        rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L305
 * \param data The input array.
 * \return new symbol
 */
inline Symbol rsqrt(Symbol data) {
  return Operator("rsqrt")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the hyperbolic sine of the input array, computed element-wise.
 *
 *        .. math::
 *        sinh(x) = 0.5\times(exp(x) - exp(-x))
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L521
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sinh(Symbol data) {
  return Operator("sinh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise exponential value of the input.
 *
 *        .. math::
 *        exp(x) = e^x \approx 2.718^x
 *
 *        Example::
 *
 *        exp([0, 1, 2]) = [inf, 1, 0.707]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L324
 * \param data The input array.
 * \return new symbol
 */
inline Symbol exp(Symbol data) {
  return Operator("exp")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the hyperbolic tangent of the input array, computed element-wise.
 *
 *        .. math::
 *        tanh(x) = sinh(x) / cosh(x)
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L549
 * \param data The input array.
 * \return new symbol
 */
inline Symbol tanh(Symbol data) {
  return Operator("tanh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise Natural logarithmic value of the input.
 *
 *        The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L334
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log(Symbol data) {
  return Operator("log")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise Base-10 logarithmic value of the input.
 *
 *        ``10**log10(x) = x``
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L344
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log10(Symbol data) {
  return Operator("log10")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the element-wise inverse hyperbolic sine of the input array, computed
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L559
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arcsinh(Symbol data) {
  return Operator("arcsinh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise Base-2 logarithmic value of the input.
 *
 *        ``2**log2(x) = x``
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L354
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log2(Symbol data) {
  return Operator("log2")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes rectified linear.
 *
 *        .. math::
 *        max(features, 0)
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L18
 * \param data The input array.
 * \return new symbol
 */
inline Symbol relu(Symbol data) {
  return Operator("relu")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the element-wise inverse hyperbolic cosine of the input array, computed
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L569
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arccosh(Symbol data) {
  return Operator("arccosh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise ``log(1 + x)`` value of the input.
 *
 *        This function is more accurate than ``log(1 + x)``  for small ``x`` so that
 *        :math:`1+x\approx 1`
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L384
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log1p(Symbol data) {
  return Operator("log1p")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes sigmoid of x element-wise.
 *
 *        .. math::
 *        y = 1 / (1 + exp(-x))
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L36
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sigmoid(Symbol data) {
  return Operator("sigmoid")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the element-wise inverse hyperbolic tangent of the input array,
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L579
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arctanh(Symbol data) {
  return Operator("arctanh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns ``exp(x) - 1`` computed element-wise on the input.
 *
 *        This function provides greater precision than ``exp(x) - 1`` for small values
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L397
 * \param data The input array.
 * \return new symbol
 */
inline Symbol expm1(Symbol data) {
  return Operator("expm1")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns the gamma function (extension of the factorial function to the reals) ,
 *
 *        From:E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:589
 * \param data The input array.
 * \return new symbol
 */
inline Symbol gamma(Symbol data) {
  return Operator("gamma")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes the element-wise sine of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L370
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sin(Symbol data) {
  return Operator("sin")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Stops gradient computation.
 *
 *        Stops the accumulated gradient of the inputs from flowing through this operator
 *        in the backward direction. In other words, this operator prevents the
 *        of its inputs to be taken into account for computing gradients.
 *
 *        Example::
 *
 *        v1 = [1, 2]
 *        v2 = [0, 1]
 *        a = Variable('a')
 *        b = Variable('b')
 *        b_stop_grad = stop_gradient(3 * b)
 *        loss = MakeLoss(b_stop_grad + a)
 *
 *        executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
 *        executor.forward(is_train=True, a=v1, b=v2)
 *        executor.outputs
 *        [ 1.  5.]
 *
 *        executor.backward()
 *        executor.grad_arrays
 *        [ 0.  0.]
 *        [ 1.  1.]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L91
 * \param data The input array.
 * \return new symbol
 */
inline Symbol BlockGrad(Symbol data) {
  return Operator("BlockGrad")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Stops gradient computation.
 *        .. note:: ``make_loss`` is deprecated, use ``MakeLoss``.
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L98
 * \param data The input array.
 * \return new symbol
 */
inline Symbol make_loss(Symbol data) {
  return Operator("make_loss")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise log of the absolute value of the gamma function of the
 *
 *        From:E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:599
 * \param data The input array.
 * \return new symbol
 */
inline Symbol gammaln(Symbol data) {
  return Operator("gammaln")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes the element-wise cosine of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L413
 * \param data The input array.
 * \return new symbol
 */
inline Symbol cos(Symbol data) {
  return Operator("cos")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Casts all elements of the input to a new type.
 *
 *        .. note:: ``Cast`` is deprecated. Use ``cast`` instead.
 *
 *        Example::
 *
 *        cast([0.9, 1.3], dtype='int32') = [0, 1]
 *        cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
 *        cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L145
 * \param data The input.
 * \param dtype Output data type.
 * \return new symbol
 */
inline Symbol Cast(Symbol data,
                   CastDtype dtype) {
  static const char *CastDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "uint8"
  };
  return Operator("Cast")
           .SetParam("dtype", CastDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Negate src
 *
 *        From:E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:164
 * \param data The input array.
 * \return new symbol
 */
inline Symbol negative(Symbol data) {
  return Operator("negative")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes the element-wise tangent of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        tan([0, \pi/4, \pi/2]) = [0, 1, -inf]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L429
 * \param data The input array.
 * \return new symbol
 */
inline Symbol tan(Symbol data) {
  return Operator("tan")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise absolute value of the input.
 *
 *        Example::
 *
 *        abs([-2, 0, 3]) = [2, 0, 3]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L176
 * \param data The input array.
 * \return new symbol
 */
inline Symbol abs(Symbol data) {
  return Operator("abs")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise inverse sine of the input array.
 *
 *        The input should be in the range `[-1, 1]`.
 *        The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].
 *
 *        .. math::
 *        arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L446
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arcsin(Symbol data) {
  return Operator("arcsin")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise sign of the input.
 *
 *        Example::
 *
 *        sign([-2, 0, 3]) = [-1, 0, 1]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L191
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sign(Symbol data) {
  return Operator("sign")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise inverse cosine of the input array.
 *
 *        The input should be in range `[-1, 1]`.
 *        The output is in the closed interval :math:`[0, \pi]`
 *
 *        .. math::
 *        arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L463
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arccos(Symbol data) {
  return Operator("arccos")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise rounded value to the nearest integer of the input.
 *
 *        Example::
 *
 *        round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L206
 * \param data The input array.
 * \return new symbol
 */
inline Symbol round(Symbol data) {
  return Operator("round")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise ceiling of the input.
 *
 *        Example::
 *
 *        ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L233
 * \param data The input array.
 * \return new symbol
 */
inline Symbol ceil(Symbol data) {
  return Operator("ceil")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise inverse tangent of the input array.
 *
 *        The output is in the closed interval :math:`[-\pi/2, \pi/2]`
 *
 *        .. math::
 *        arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L479
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arctan(Symbol data) {
  return Operator("arctan")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise floor of the input.
 *
 *        Example::
 *
 *        floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L244
 * \param data The input array.
 * \return new symbol
 */
inline Symbol floor(Symbol data) {
  return Operator("floor")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise rounded value to the nearest integer of the input.
 *
 *        .. note::
 *        - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.
 *        - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.
 *
 *        Example::
 *
 *        rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L222
 * \param data The input array.
 * \return new symbol
 */
inline Symbol rint(Symbol data) {
  return Operator("rint")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Converts each element of the input array from radians to degrees.
 *
 *        .. math::
 *        degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L493
 * \param data The input array.
 * \return new symbol
 */
inline Symbol degrees(Symbol data) {
  return Operator("degrees")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Returns element-wise rounded value to the nearest integer towards zero of the
 *
 *        Example::
 *
 *        fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\elemwise_unary_op.cc:L255
 * \param data The input array.
 * \return new symbol
 */
inline Symbol fix(Symbol data) {
  return Operator("fix")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Slices a contiguous region of the array.
 *
 *        .. note:: ``crop`` is deprecated. Use ``slice`` instead.
 *
 *        This function returns a sliced continous region of the array between the
 *        by `begin` and `end`.
 *
 *        For an input array of `n` dimensions, slice operation with ``begin=(b_0,
 *        and ``end=(e_1, e_2, ... e_n)`` indices will result in an array with the shape
 *        ``(e_1-b_0, ..., e_n-b_n-1)``.
 *
 *        The resulting array's *k*-th dimension contains elements
 *        from the *k*-th dimension of the input array with the open range ``[b_k, e_k)``.
 *
 *        Example::
 *
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
 *        [ 6.,  7.,  8.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L244
 * \param data Source input
 * \param begin starting indices for the slice operation, supports negative indices.
 * \param end ending indices for the slice operation, supports negative indices.
 * \return new symbol
 */
inline Symbol slice(Symbol data,
                    Shape begin,
                    Shape end) {
  return Operator("slice")
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Slices along a given axis.
 *
 *        Returns an array slice along a given `axis` starting from the `begin` index
 *        to the `end` index.
 *
 *        Examples::
 *
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *
 *        slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],
 *        [  5.,   6.],
 *        [  9.,  10.]]
 *
 *        slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],
 *        [  6.,   7.],
 *        [ 10.,  11.]]
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L324
 * \param data Source input
 * \param axis Axis along which to be sliced, supports negative indexes.
 * \param begin The beginning index along the axis to be sliced,  supports negative
 * \param end The ending index along the axis to be sliced,  supports negative indexes.
 * \return new symbol
 */
inline Symbol slice_axis(Symbol data,
                         int axis,
                         int begin,
                         dmlc::optional<int> end) {
  return Operator("slice_axis")
           .SetParam("axis", axis)
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Dot product of two arrays.
 *
 *        ``dot``'s behavior depends on the input array dimensions:
 *
 *        - 1-D arrays: inner product of vectors
 *        - 2-D arrays: matrix multiplication
 *        - N-D arrays: a sum product over the last axis of the first input and the first
 *        axis of the second input
 *
 *        For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape
 *        result array will have shape `(n,m,r,s)`. It is computed by::
 *
 *        dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L357
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \return new symbol
 */
inline Symbol dot(Symbol lhs,
                  Symbol rhs,
                  bool transpose_a = false,
                  bool transpose_b = false) {
  return Operator("dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Batchwise dot product.
 *
 *        ``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
 *        ``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.
 *
 *        For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape
 *        `(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,
 *        which is computed by::
 *
 *        batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L393
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \return new symbol
 */
inline Symbol batch_dot(Symbol lhs,
                        Symbol rhs,
                        bool transpose_a = false,
                        bool transpose_b = false) {
  return Operator("batch_dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \breif Clips (limits) the values in an array.
 *
 *        Given an interval, values outside the interval are clipped to the interval
 *        Clipping ``x`` between `a_min` and `a_x` would be::
 *
 *        clip(x, a_min, a_max) = max(min(x, a_max), a_min))
 *
 *        Example::
 *
 *        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 *
 *        clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L438
 * \param data Input array.
 * \param a_min Minimum value
 * \param a_max Maximum value
 * \return new symbol
 */
inline Symbol clip(Symbol data,
                   mx_float a_min,
                   mx_float a_max) {
  return Operator("clip")
           .SetParam("a_min", a_min)
           .SetParam("a_max", a_max)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Repeats elements of an array.
 *
 *        By default, ``repeat`` flattens the input array into 1-D and then repeats the
 *        elements::
 *
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *
 *        repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]
 *
 *        The parameter ``axis`` specifies the axis along which to perform repeat::
 *
 *        repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],
 *        [ 3.,  3.,  4.,  4.]]
 *
 *        repeat(x, repeats=2, axis=0) = [[ 1.,  2.],
 *        [ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 3.,  4.]]
 *
 *        repeat(x, repeats=2, axis=-1) = [[ 1.,  1.,  2.,  2.],
 *        [ 3.,  3.,  4.,  4.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L480
 * \param data Input data array
 * \param repeats The number of repetitions for each element.
 * \param axis The axis along which to repeat values. The negative numbers are
 *        interpreted counting from the backward. By default, use the flattened input
 * \return new symbol
 */
inline Symbol repeat(Symbol data,
                     int repeats,
                     dmlc::optional<int> axis = dmlc::optional<int>()) {
  return Operator("repeat")
           .SetParam("repeats", repeats)
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Repeats the whole array multiple times.
 *
 *        If ``reps`` has length *d*, and input array has dimension of *n*. There are
 *        there cases:
 *
 *        - **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::
 *
 *        x = [[1, 2],
 *        [3, 4]]
 *
 *        tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]
 *
 *        - **n>d**. ``reps`` is promoted to length *n* by pre-pending 1's to it. Thus for
 *        an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::
 *
 *
 *        tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.]]
 *
 *        - **n<d**. The input is promoted to be d-dimensional by prepending new axes. So
 *        shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::
 *
 *        tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]],
 *
 *        [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]]
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L537
 * \param data Input data array
 * \param reps The number of times for repeating the tensor a. If reps has length d, the
 *        result will have dimension of max(d, a.ndim); If a.ndim < d, a is promoted to
 *        be d-dimensional by prepending new axes. If a.ndim > d, reps is promoted to
 * \return new symbol
 */
inline Symbol tile(Symbol data,
                   Shape reps) {
  return Operator("tile")
           .SetParam("reps", reps)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Reverses the order of elements along given axis while preserving array shape.
 *
 *        Note: reverse and flip are equivalent. We use reverse in the following examples.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.,  3.,  4.],
 *        [ 5.,  6.,  7.,  8.,  9.]]
 *
 *        reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],
 *        [ 0.,  1.,  2.,  3.,  4.]]
 *
 *        reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],
 *        [ 9.,  8.,  7.,  6.,  5.]]
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L574
 * \param data Input data array
 * \param axis The axis which to reverse elements.
 * \return new symbol
 */
inline Symbol reverse(Symbol data,
                      Shape axis) {
  return Operator("reverse")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Reshapes the input array.
 *
 *        .. note:: ``Reshape`` is deprecated, use ``reshape``
 *
 *        Given an array and a shape, this function returns a copy of the array in the
 *        The shape is a tuple of integers such as (2,3,4).The size of the new shape
 *
 *        Example::
 *
 *        reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]
 *
 *        Some dimensions of the shape can take special values from the set {0, -1, -2,
 *
 *        - ``0``  copy this dimension from the input to the output shape.
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
 *        - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)
 *
 *        - ``-1`` infers the dimension of the output shape by using the remainder of the
 *        keeping the size of the new array same as that of the input array.
 *        At most one dimension of shape can be -1.
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
 *        - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
 *        - input shape = (2,3,4), shape=(-1,), output shape = (24,)
 *
 *        - ``-2`` copy all/remainder of the input dimensions to the output shape.
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
 *        - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
 *        - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)
 *
 *        - ``-3`` use the product of two consecutive dimensions of the input shape as
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
 *        - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
 *        - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
 *        - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)
 *
 *        - ``-4`` split one dimension of the input into two dimensions passed subsequent
 *
 *        Example::
 *
 *        - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
 *        - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)
 *
 *        If the argument `reverse` is set to 1, then the special values are inferred
 *
 *        Example::
 *
 *        - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape
 *        - with reverse=1, output shape will be (50,4).
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L87
 * \param data Input data to reshape.
 * \param shape The target shape
 * \param reverse If true then the special values are inferred from right to left
 * \param target_shape (Deprecated! Use ``shape`` instead.) Target new shape. One and
 * \param keep_highest (Deprecated! Use ``shape`` instead.) Whether keep the highest dim
 *        unchanged.If set to true, then the first dim in target_shape is ignored,and
 * \return new symbol
 */
inline Symbol Reshape(Symbol data,
                      Shape shape = Shape(),
                      bool reverse = false,
                      Shape target_shape = Shape(0,0),
                      bool keep_highest = false) {
  return Operator("Reshape")
           .SetParam("shape", shape)
           .SetParam("reverse", reverse)
           .SetParam("target_shape", target_shape)
           .SetParam("keep_highest", keep_highest)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Flattens the input array into a 2-D array by collapsing the higher dimensions.
 *
 *        .. note:: `Flatten` is deprecated. Use `flatten` instead.
 *
 *        For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation
 *        the input array into an output array of shape ``(d1, d2*...*dk)``.
 *
 *        Example::
 *
 *        x = [[
 *        [1,2,3],
 *        [4,5,6],
 *        [7,8,9]
 *        ],
 *        [    [1,2,3],
 *        [4,5,6],
 *        [7,8,9]
 *        ]],
 *
 *        flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
 *        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L127
 * \param data Input array.
 * \return new symbol
 */
inline Symbol Flatten(Symbol data) {
  return Operator("Flatten")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Permutes the dimensions of an array.
 *
 *        Examples::
 *
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *
 *        transpose(x) = [[ 1.,  3.],
 *        [ 2.,  4.]]
 *
 *        x = [[[ 1.,  2.],
 *        [ 3.,  4.]],
 *
 *        [[ 5.,  6.],
 *        [ 7.,  8.]]]
 *
 *        transpose(x) = [[[ 1.,  5.],
 *        [ 3.,  7.]],
 *
 *        [[ 2.,  6.],
 *        [ 4.,  8.]]]
 *
 *        transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],
 *        [ 5.,  6.]],
 *
 *        [[ 3.,  4.],
 *        [ 7.,  8.]]]
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L168
 * \param data Source input
 * \param axes Target axis order. By default the axes will be inverted.
 * \return new symbol
 */
inline Symbol transpose(Symbol data,
                        Shape axes = Shape()) {
  return Operator("transpose")
           .SetParam("axes", axes)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Inserts a new axis of size 1 into the array shape
 *
 *        For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``
 *        will return a new array with shape ``(2,1,3,4)``.
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\tensor\matrix_op.cc:L204
 * \param data Source input
 * \param axis Position (amongst axes) where new axis is to be inserted.
 * \return new symbol
 */
inline Symbol expand_dims(Symbol data,
                          uint32_t axis) {
  return Operator("expand_dims")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Batch normalization.
 *
 *        Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
 *        well as offset ``beta``.
 *
 *        Assume the input has more than one dimension and we normalize along axis 1.
 *        We first compute the mean and variance along this axis:
 *
 *        .. math::
 *
 *        data\_mean[i] = mean(data[:,i,:,...]) \\
 *        data\_var[i] = var(data[:,i,:,...])
 *
 *        Then compute the normalized output, which has the same shape as input, as
 *
 *        .. math::
 *
 *        out[:,i,:,...] = \frac{data[:,i,:,...] -
 *
 *        Both *mean* and *var* returns a scalar by treating the input as a vector.
 *
 *        Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
 *        have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both
 *        ``data_var`` as well, which are needed for the backward pass.
 *
 *        Besides the inputs and the outputs, this operator accepts two auxiliary
 *        states, ``moving_mean`` and ``moving_var``, which are *k*-length
 *        vectors. They are global statistics for the whole dataset, which are updated
 *        by::
 *
 *        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
 *        moving_var = moving_var * momentum + data_var * (1 - momentum)
 *
 *        If ``use_global_stats`` is set to be true, then ``moving_mean`` and
 *        ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
 *        the output. It is often used during inference.
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is
 *        then set ``gamma`` to 1 and its gradient to 0.
 *
 *
 *
 *        Defined in
 * \param data Input data to batch normalization
 * \param gamma gamma array
 * \param beta beta array
 * \param eps Epsilon to prevent div 0. Must be bigger than CUDNN_BN_MIN_EPSILON defined
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of local
 * \param output_mean_var Output All,normal mean and var
 * \return new symbol
 */
inline Symbol BatchNorm(Symbol data,
                        Symbol gamma,
                        Symbol beta,
                        mx_float eps = 0.001,
                        mx_float momentum = 0.9,
                        bool fix_gamma = true,
                        bool use_global_stats = false,
                        bool output_mean_var = false) {
  return Operator("BatchNorm")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetParam("output_mean_var", output_mean_var)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol();
}

/*!
 * \breif Joins input arrays along a given axis.
 *
 *        .. note:: `Concat` is deprecated. Use `concat` instead.
 *
 *        The dimensions of the input arrays should be the same except the axis along
 *        which they will concatenated.
 *        The dimension of the output array along the concatenated axis will be equal
 *        to the sum of the corresponding dimensions of the input arrays.
 *
 *        Example::
 *
 *        x = [[1,1],[2,2]]
 *        y = [[3,3],[4,4],[5,5]]
 *        z = [[6,6], [7,7],[8,8]]
 *
 *        concat(x,y,z,dim=0) = [[ 1.,  1.],
 *        [ 2.,  2.],
 *        [ 3.,  3.],
 *        [ 4.,  4.],
 *        [ 5.,  5.],
 *        [ 6.,  6.],
 *        [ 7.,  7.],
 *        [ 8.,  8.]]
 *
 *        Note that you cannot concat x,y,z along dimension 1 since dimension
 *        0 is not the same for all the input arrays.
 *
 *        concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
 *        [ 4.,  4.,  7.,  7.],
 *        [ 5.,  5.,  8.,  8.]]
 *
 *
 *
 *        Defined in
 * \param data List of arrays to concatenate
 * \param num_args Number of inputs to be concated.
 * \param dim the dimension to be concated.
 * \return new symbol
 */
inline Symbol Concat(const std::vector<Symbol>& data,
                     int num_args,
                     int dim = 1) {
  return Operator("Concat")
           .SetParam("num_args", num_args)
           .SetParam("dim", dim)
(data)
           .CreateSymbol();
}

/*!
 * \breif Apply a sparse regularization to the output a sigmoid activation function.
 * \param data Input data.
 * \param sparseness_target The sparseness target
 * \param penalty The tradeoff parameter for the sparseness penalty
 * \param momentum The momentum for running average
 * \return new symbol
 */
inline Symbol IdentityAttachKLSparseReg(Symbol data,
                                        mx_float sparseness_target = 0.1,
                                        mx_float penalty = 0.001,
                                        mx_float momentum = 0.9) {
  return Operator("IdentityAttachKLSparseReg")
           .SetParam("sparseness_target", sparseness_target)
           .SetParam("penalty", penalty)
           .SetParam("momentum", momentum)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Applies Leaky rectified linear unit activation element-wise to the input.
 *
 *        Leaky ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope`
 *        when the input is negative and has a slope of one when input is positive.
 *
 *        The following modified ReLU Activation functions are supported:
 *
 *        - *elu*: Exponential Linear Unit. `y = x > 0 ? x : slope * (exp(x)-1)`
 *        - *leaky*: Leaky ReLU. `y = x > 0 ? x : slope * x`
 *        - *prelu*: Parametric ReLU. This is same as *leaky* except that `slope` is
 *        - *rrelu*: Randomized ReLU. same as *leaky* but the `slope` is uniformly and
 *        *[lower_bound, upper_bound)* for training, while fixed to be
 *        *(lower_bound+upper_bound)/2* for inference.
 *
 *
 *
 *        Defined in
 * \param data Input data to activation function.
 * \param act_type Activation function to be applied.
 * \param slope Init slope for the activation. (For leaky and elu only)
 * \param lower_bound Lower bound of random slope. (For rrelu only)
 * \param upper_bound Upper bound of random slope. (For rrelu only)
 * \return new symbol
 */
inline Symbol LeakyReLU(Symbol data,
                        LeakyReLUActType act_type = LeakyReLUActType::leaky,
                        mx_float slope = 0.25,
                        mx_float lower_bound = 0.125,
                        mx_float upper_bound = 0.334) {
  static const char *LeakyReLUActTypeValues[] = {
    "elu",
    "leaky",
    "prelu",
    "rrelu"
  };
  return Operator("LeakyReLU")
           .SetParam("act_type", LeakyReLUActTypeValues[int(act_type)])
           .SetParam("slope", slope)
           .SetParam("lower_bound", lower_bound)
           .SetParam("upper_bound", upper_bound)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Pads an input array with a constant or edge values of the array.
 *
 *        .. note:: `Pad` is deprecated. Use `pad` instead.
 *
 *        .. note:: Current implementation only supports 4D and 5D input arrays with
 *        only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.
 *
 *        This operation pads an input array with either a `constant_value` or edge values
 *        along each axis of the input array. The amount of padding is specified by
 *
 *        `pad_width` is a tuple of integer padding widths for each axis of the format
 *        ``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of
 *        where ``N`` is the number of dimensions of the array.
 *
 *        For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates
 *        to add before and after the elements of the array along dimension ``N``.
 *        The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,
 *        ``after_2`` must be 0.
 *
 *        Example::
 *
 *        x = [[[[  1.   2.   3.]
 *        [  4.   5.   6.]]
 *
 *        [[  7.   8.   9.]
 *        [ 10.  11.  12.]]]
 *
 *
 *        [[[ 11.  12.  13.]
 *        [ 14.  15.  16.]]
 *
 *        [[ 17.  18.  19.]
 *        [ 20.  21.  22.]]]]
 *
 *        pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =
 *
 *        [[[[  1.   1.   2.   3.   3.]
 *        [  1.   1.   2.   3.   3.]
 *        [  4.   4.   5.   6.   6.]
 *        [  4.   4.   5.   6.   6.]]
 *
 *        [[  7.   7.   8.   9.   9.]
 *        [  7.   7.   8.   9.   9.]
 *        [ 10.  10.  11.  12.  12.]
 *        [ 10.  10.  11.  12.  12.]]]
 *
 *
 *        [[[ 11.  11.  12.  13.  13.]
 *        [ 11.  11.  12.  13.  13.]
 *        [ 14.  14.  15.  16.  16.]
 *        [ 14.  14.  15.  16.  16.]]
 *
 *        [[ 17.  17.  18.  19.  19.]
 *        [ 17.  17.  18.  19.  19.]
 *        [ 20.  20.  21.  22.  22.]
 *        [ 20.  20.  21.  22.  22.]]]]
 *
 *        pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,2,2,1,1)) =
 *
 *        [[[[  0.   0.   0.   0.   0.]
 *        [  0.   1.   2.   3.   0.]
 *        [  0.   4.   5.   6.   0.]
 *        [  0.   0.   0.   0.   0.]]
 *
 *        [[  0.   0.   0.   0.   0.]
 *        [  0.   7.   8.   9.   0.]
 *        [  0.  10.  11.  12.   0.]
 *        [  0.   0.   0.   0.   0.]]]
 *
 *
 *        [[[  0.   0.   0.   0.   0.]
 *        [  0.  11.  12.  13.   0.]
 *        [  0.  14.  15.  16.   0.]
 *        [  0.   0.   0.   0.   0.]]
 *
 *        [[  0.   0.   0.   0.   0.]
 *        [  0.  17.  18.  19.   0.]
 *        [  0.  20.  21.  22.   0.]
 *        [  0.   0.   0.   0.   0.]]]]
 *
 *
 *
 *
 *        Defined in
 * \param data An n-dimensional input array.
 * \param mode Padding type to use. "constant" pads with `constant_value` and "edge" pads
 * \param pad_width Widths of the padding regions applied to the edges of each axis. It
 *        is a tuple of integer padding widths for each axis of the format ``(before_1,
 *        after_1, ... , before_N, after_N)``. It should be of length ``2*N`` where ``N``
 *        is the number of dimensions of the array.This is equivalent to pad_width in
 * \param constant_value The value used for padding when `mode` is "constant".
 * \return new symbol
 */
inline Symbol Pad(Symbol data,
                  PadMode mode,
                  Shape pad_width,
                  double constant_value = 0) {
  static const char *PadModeValues[] = {
    "constant",
    "edge"
  };
  return Operator("Pad")
           .SetParam("mode", PadModeValues[int(mode)])
           .SetParam("pad_width", pad_width)
           .SetParam("constant_value", constant_value)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Splits an array along a particular axis into multiple sub-arrays.
 *
 *        .. note:: ``SliceChannel`` is depreacted. Use ``split`` instead.
 *
 *        **Note** that `num_outputs` should evenly divide the length of the axis
 *        along which to split the array.
 *
 *        Example::
 *
 *        x  = [[[ 1.]
 *        [ 2.]]
 *        [[ 3.]
 *        [ 4.]]
 *        [[ 5.]
 *        [ 6.]]]
 *        x.shape = (3, 2, 1)
 *
 *        y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)
 *        y = [[[ 1.]]
 *        [[ 3.]]
 *        [[ 5.]]]
 *
 *        [[[ 2.]]
 *        [[ 4.]]
 *        [[ 6.]]]
 *
 *        y[0].shape = (3, 1, 1)
 *
 *        z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)
 *        z = [[[ 1.]
 *        [ 2.]]]
 *
 *        [[[ 3.]
 *        [ 4.]]]
 *
 *        [[[ 5.]
 *        [ 6.]]]
 *
 *        z[0].shape = (1, 2, 1)
 *
 *        `squeeze_axis=1` removes the axis with length 1 from the shapes of the output
 *        **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
 *        along the `axis` which it is split.
 *        Also `squeeze_axis` can be set to true only if ``input.shape[axis] ==
 *
 *        z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with
 *        z = [[ 1.]
 *        [ 2.]]
 *
 *        [[ 3.]
 *        [ 4.]]
 *
 *        [[ 5.]
 *        [ 6.]]
 *        z[0].shape = (2 ,1 )
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\slice_channel.cc:L86
 * \param data The input
 * \param num_outputs Number of splits. Note that this should evenly divide the length of
 * \param axis Axis along which to split.
 * \param squeeze_axis If true, Removes the axis with length 1 from the shapes of the
 *        output arrays. **Note** that setting `squeeze_axis` to ``true`` removes axis
 *        with length 1 only along the `axis` which it is split. Also `squeeze_axis` can
 * \return new symbol
 */
inline Symbol SliceChannel(Symbol data,
                           int num_outputs,
                           int axis = 1,
                           bool squeeze_axis = false) {
  return Operator("SliceChannel")
           .SetParam("num_outputs", num_outputs)
           .SetParam("axis", axis)
           .SetParam("squeeze_axis", squeeze_axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Interchanges two axes of an array.
 *
 *        Examples::
 *
 *        x = [[1, 2, 3]])
 *        swapaxes(x, 0, 1) = [[ 1],
 *        [ 2],
 *        [ 3]]
 *
 *        x = [[[ 0, 1],
 *        [ 2, 3]],
 *        [[ 4, 5],
 *        [ 6, 7]]]  // (2,2,2) array
 *
 *        swapaxes(x, 0, 2) = [[[ 0, 4],
 *        [ 2, 6]],
 *        [[ 1, 5],
 *        [ 3, 7]]]
 *
 *
 *        Defined in
 * \param data Input array.
 * \param dim1 the first axis to be swapped.
 * \param dim2 the second axis to be swapped.
 * \return new symbol
 */
inline Symbol SwapAxis(Symbol data,
                       uint32_t dim1 = 0,
                       uint32_t dim2 = 0) {
  return Operator("SwapAxis")
           .SetParam("dim1", dim1)
           .SetParam("dim2", dim2)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Performs nearest neighbor/bilinear up sampling to inputs
 * \param data Array of tensors to upsample
 * \param scale Up sampling scale
 * \param sample_type upsampling method
 * \param num_args Number of inputs to be upsampled. For nearest neighbor upsampling,
 *        this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other
 *        inputs will be upsampled to thesame size. For bilinear upsampling this must be
 * \param num_filter Input filter. Only used by bilinear sample_type.
 * \param multi_input_mode How to handle multiple input. concat means concatenate
 *        upsampled images along the channel dimension. sum means add all images
 * \param workspace Tmp workspace for deconvolution (MB)
 * \return new symbol
 */
inline Symbol UpSampling(const std::vector<Symbol>& data,
                         uint32_t scale,
                         UpSamplingSampleType sample_type,
                         int num_args,
                         uint32_t num_filter = 0,
                         UpSamplingMultiInputMode multi_input_mode = UpSamplingMultiInputMode::concat,
                         uint64_t workspace = 512) {
  static const char *UpSamplingSampleTypeValues[] = {
    "bilinear",
    "nearest"
  };
  static const char *UpSamplingMultiInputModeValues[] = {
    "concat",
    "sum"
  };
  return Operator("UpSampling")
           .SetParam("scale", scale)
           .SetParam("sample_type", UpSamplingSampleTypeValues[int(sample_type)])
           .SetParam("num_args", num_args)
           .SetParam("num_filter", num_filter)
           .SetParam("multi_input_mode", UpSamplingMultiInputModeValues[int(multi_input_mode)])
           .SetParam("workspace", workspace)
(data)
           .CreateSymbol();
}

/*!
 * \breif Applies an activation function element-wise to the input.
 *
 *        The following activation functions are supported:
 *
 *        - `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
 *        - `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
 *        - `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) +
 *        - `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
 *
 *
 *
 *        Defined in
 * \param data Input array to activation function.
 * \param act_type Activation function to be applied.
 * \return new symbol
 */
inline Symbol Activation(Symbol data,
                         ActivationActType act_type) {
  static const char *ActivationActTypeValues[] = {
    "relu",
    "sigmoid",
    "softrelu",
    "tanh"
  };
  return Operator("Activation")
           .SetParam("act_type", ActivationActTypeValues[int(act_type)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Applies bilinear sampling to input feature map, which is the key of "[NIPS2015]
 *        output[batch, channel, y_dst, x_dst] = G(data[batch, channel, y_src, x_src)
 *        x_dst, y_dst enumerate all spatial locations in output
 *        x_src = grid[batch, 0, y_dst, x_dst]
 *        y_src = grid[batch, 1, y_dst, x_dst]
 *        G() denotes the bilinear interpolation kernel
 *        The out-boundary points will be padded as zeros. (The boundary is defined to be
 *        The shape of output will be (data.shape[0], data.shape[1], grid.shape[2],
 *        The operator assumes that grid has been nomalized. If you want to design a
 * \param data Input data to the BilinearsamplerOp.
 * \param grid Input grid to the BilinearsamplerOp.grid has two channels: x_src, y_src
 * \return new symbol
 */
inline Symbol BilinearSampler(Symbol data,
                              Symbol grid) {
  return Operator("BilinearSampler")
           .SetInput("data", data)
           .SetInput("grid", grid)
           .CreateSymbol();
}

/*!
 * \breif Compute *N*-D convolution on *(N+2)*-D input.
 *
 *        In the 2-D convolution, given input data with shape *(batch_size,
 *        channel, height, width)*, the output is computed by
 *
 *        .. math::
 *
 *        out[n,i,:,:] = bias[i] + \sum_{j=0}^{num\_filter} data[n,j,:,:] \star
 *        weight[i,j,:,:]
 *
 *        where :math:`\star` is the 2-D cross-correlation operator.
 *
 *        For general 2-D convolution, the shapes are
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*.
 *
 *        Define::
 *
 *        f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1
 *
 *        then we have::
 *
 *        out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
 *        out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *        The default data ``layout`` is *NCHW*, namely *(batch_size, channle, height,
 *        width)*. We can choose other layouts such as *NHWC*.
 *
 *        If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
 *        evenly into *g* parts along the channel axis, and also evenly split ``weight``
 *        along the first dimension. Next compute the convolution on the *i*-th part of
 *        the data with the *i*-th weight part. The output is obtained by concating all
 *        the *g* results.
 *
 *        1-D convolution does not have *height* dimension but only *width* in space.
 *
 *        - **data**: *(batch_size, channel, width)*
 *        - **weight**: *(num_filter, channel, kernel[0])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_width)*.
 *
 *        3-D convolution adds an additional *depth* dimension besides *height* and
 *        *width*. The shapes are
 *
 *        - **data**: *(batch_size, channel, depth, height, width)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.
 *
 *        Both ``weight`` and ``bias`` are learnable parameters.
 *
 *        There are other options to tune the performance.
 *
 *        - **cudnn_tune**: enable this option leads to higher startup time but may give
 *        faster speed. Options are
 *
 *        - **off**: no tuning
 *        - **limited_workspace**:run test and pick the fastest algorithm that doesn't
 *        exceed workspace limit.
 *        - **fastest**: pick the fastest algorithm and ignore workspace limit.
 *        - **None** (default): the behavior is determined by environment variable
 *        ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace
 *        (default), 2 for fastest.
 *
 *        - **workspace**: A large number leads to more (GPU) memory usage but may improve
 *        the performance.
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\convolution.cc:L154
 * \param data Input data to the ConvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel convolution kernel size: (h, w) or (d, h, w)
 * \param num_filter convolution filter(channel) number
 * \param stride convolution stride: (h, w) or (d, h, w)
 * \param dilate convolution dilate: (h, w) or (d, h, w)
 * \param pad pad for convolution: (h, w) or (d, h, w)
 * \param num_group Number of group partitions.
 * \param workspace Maximum temperal workspace allowed for convolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Convolution(Symbol data,
                          Symbol weight,
                          Symbol bias,
                          Shape kernel,
                          uint32_t num_filter,
                          Shape stride = Shape(),
                          Shape dilate = Shape(),
                          Shape pad = Shape(),
                          uint32_t num_group = 1,
                          uint64_t workspace = 1024,
                          bool no_bias = false,
                          ConvolutionCudnnTune cudnn_tune = ConvolutionCudnnTune::None,
                          bool cudnn_off = false,
                          ConvolutionLayout layout = ConvolutionLayout::None) {
  static const char *ConvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *ConvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NCW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", ConvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", ConvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \breif This operator is DEPRECATED. Apply convolution to input then add a bias.
 * \param data Input data to the ConvolutionV1Op.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel convolution kernel size: (h, w) or (d, h, w)
 * \param num_filter convolution filter(channel) number
 * \param stride convolution stride: (h, w) or (d, h, w)
 * \param dilate convolution dilate: (h, w) or (d, h, w)
 * \param pad pad for convolution: (h, w) or (d, h, w)
 * \param num_group Number of group partitions. Equivalent to slicing input into num_group
 *        partitions, apply convolution on each, then concatenate the results
 * \param workspace Maximum tmp workspace allowed for convolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 *        Leads to higher startup time but may give faster speed. Options are:
 *        'off': no tuning
 *        'limited_workspace': run test and pick the fastest algorithm that doesn't
 *        'fastest': pick the fastest algorithm and ignore workspace limit.
 *        If set to None (default), behavior is determined by environment
 *        variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,
 *        1 for limited workspace (default), 2 for fastest.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Convolution_v1(Symbol data,
                             Symbol weight,
                             Symbol bias,
                             Shape kernel,
                             uint32_t num_filter,
                             Shape stride = Shape(),
                             Shape dilate = Shape(),
                             Shape pad = Shape(),
                             uint32_t num_group = 1,
                             uint64_t workspace = 1024,
                             bool no_bias = false,
                             Convolution_v1CudnnTune cudnn_tune = Convolution_v1CudnnTune::None,
                             bool cudnn_off = false,
                             Convolution_v1Layout layout = Convolution_v1Layout::None) {
  static const char *Convolution_v1CudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *Convolution_v1LayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution_v1")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", Convolution_v1CudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", Convolution_v1LayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \breif Applies correlation to inputs.
 * \param data1 Input data1 to the correlation.
 * \param data2 Input data2 to the correlation.
 * \param kernel_size kernel size for Correlation must be an odd number
 * \param max_displacement Max displacement of Correlation
 * \param stride1 stride1 quantize data1 globally
 * \param stride2 stride2 quantize data2 within the neighborhood centered around data1
 * \param pad_size pad for Correlation
 * \param is_multiply operation type is either multiplication or subduction
 * \return new symbol
 */
inline Symbol Correlation(Symbol data1,
                          Symbol data2,
                          uint32_t kernel_size = 1,
                          uint32_t max_displacement = 1,
                          uint32_t stride1 = 1,
                          uint32_t stride2 = 1,
                          uint32_t pad_size = 0,
                          bool is_multiply = true) {
  return Operator("Correlation")
           .SetParam("kernel_size", kernel_size)
           .SetParam("max_displacement", max_displacement)
           .SetParam("stride1", stride1)
           .SetParam("stride2", stride2)
           .SetParam("pad_size", pad_size)
           .SetParam("is_multiply", is_multiply)
           .SetInput("data1", data1)
           .SetInput("data2", data2)
           .CreateSymbol();
}

/*!
 * \breif
 *
 *        .. note:: `Crop` is deprecated. Use `slice` instead.
 *
 *        Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or
 *        with width and height of the second input symbol, i.e., with one input, we need
 *        specify the crop height and width, otherwise the second input symbol's size
 *
 *
 *        Defined in
 * \param data Tensor or List of Tensors, the second input will be used as crop_like
 * \param num_args Number of inputs for crop, if equals one, then we will use the h_wfor
 *        crop height and width, else if equals two, then we will use the heightand width
 * \param offset crop offset coordinate: (y, x)
 * \param h_w crop height and width: (h, w)
 * \param center_crop If set to true, then it will use be the center_crop,or it will crop
 * \return new symbol
 */
inline Symbol Crop(const std::vector<Symbol>& data,
                   int num_args,
                   Shape offset = Shape(0,0),
                   Shape h_w = Shape(0,0),
                   bool center_crop = false) {
  return Operator("Crop")
           .SetParam("num_args", num_args)
           .SetParam("offset", offset)
           .SetParam("h_w", h_w)
           .SetParam("center_crop", center_crop)
(data)
           .CreateSymbol();
}

/*!
 * \breif Applies deconvolution to input and adds a bias.
 * \param data Input data to the DeconvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel deconvolution kernel size: (h, w) or (d, h, w)
 * \param num_filter deconvolution filter(channel) number
 * \param stride deconvolution stride: (h, w) or (d, h, w)
 * \param dilate deconvolution dilate: (h, w) or (d, h, w)
 * \param pad pad for deconvolution: (h, w) or (d, h, w). A good number is :
 *        (kernel-1)/2. If target_shape is set, pad will be ignored and computed
 * \param adj adjustment for output shape: (h, w) or (d, h, w). If target_shape is set,
 * \param target_shape output shape with target shape : (h, w) or (d, h, w)
 * \param num_group number of groups partition
 * \param workspace Maximum temporal workspace allowed for deconvolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Deconvolution(Symbol data,
                            Symbol weight,
                            Symbol bias,
                            Shape kernel,
                            uint32_t num_filter,
                            Shape stride = Shape(),
                            Shape dilate = Shape(),
                            Shape pad = Shape(),
                            Shape adj = Shape(),
                            Shape target_shape = Shape(),
                            uint32_t num_group = 1,
                            uint64_t workspace = 512,
                            bool no_bias = true,
                            DeconvolutionCudnnTune cudnn_tune = DeconvolutionCudnnTune::None,
                            bool cudnn_off = false,
                            DeconvolutionLayout layout = DeconvolutionLayout::None) {
  static const char *DeconvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *DeconvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NCW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Deconvolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("adj", adj)
           .SetParam("target_shape", target_shape)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", DeconvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", DeconvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \breif Applies dropout to input.
 *        During training, each element of the input is randomly set to zero with
 *        And then the whole tensor is rescaled by 1/(1-p) to keep the expectation the
 *        before applying dropout. During the test time, this behaves as an identity map.
 *
 * \param data Input data to dropout.
 * \param p Fraction of the input that gets dropped out at training time
 * \return new symbol
 */
inline Symbol Dropout(Symbol data,
                      mx_float p = 0.5) {
  return Operator("Dropout")
           .SetParam("p", p)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Applies a linear transformation: :math:`Y = XW^T + b`.
 *
 *        Shapes:
 *
 *        - **data**: `(batch_size, input_dim)`
 *        - **weight**: `(num_hidden, input_dim)`
 *        - **bias**: `(num_hidden,)`
 *        - **out**: `(batch_size, num_hidden)`
 *
 *        The learnable parameters include both ``weight`` and ``bias``.
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\fully_connected.cc:L74
 * \param data Input data.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param num_hidden Number of hidden nodes of the output.
 * \param no_bias Whether to disable bias parameter.
 * \return new symbol
 */
inline Symbol FullyConnected(Symbol data,
                             Symbol weight,
                             Symbol bias,
                             int num_hidden,
                             bool no_bias = false) {
  return Operator("FullyConnected")
           .SetParam("num_hidden", num_hidden)
           .SetParam("no_bias", no_bias)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \breif Generates sampling grid for bilinear sampling.
 * \param data Input data to the GridGeneratorOp.
 * \param transform_type transformation type
 *        if transformation type is affine, data is affine matrix : (batch, 6)
 *        if transformation type is warp, data is optical flow : (batch, 2, h, w)
 * \param target_shape if transformation type is affine, the operator need a target_shape
 *        if transofrmation type is warp, the operator will ignore target_shape
 * \return new symbol
 */
inline Symbol GridGenerator(Symbol data,
                            GridGeneratorTransformType transform_type,
                            Shape target_shape = Shape(0,0)) {
  static const char *GridGeneratorTransformTypeValues[] = {
    "affine",
    "warp"
  };
  return Operator("GridGenerator")
           .SetParam("transform_type", GridGeneratorTransformTypeValues[int(transform_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Applies instance normalization to the n-dimensional input array.
 *
 *        This operator takes an n-dimensional input array where (n>2) and normalizes
 *        the input using the following formula:
 *
 *        .. math::
 *
 *        out = \frac{x - mean[data]}{ \sqrt{Var[data]} + \epsilon} * gamma + beta
 *
 *        This layer is similar to batch normalization layer (`BatchNorm`)
 *        with two differences: first, the normalization is
 *        carried out per example (instance), not over a batch. Second, the
 *        same normalization is applied both at test and train time. This
 *        operation is also known as `contrast normalization`.
 *
 *        If the input data is of shape [batch, channel, spacial_dim1, spacial_dim2, ...],
 *        `gamma` and `beta` parameters must be vectors of shape [channel].
 *
 *        This implementation is based on paper:
 *
 *        .. [1] Instance Normalization: The Missing Ingredient for Fast Stylization,
 *        D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2).
 *
 *        Examples::
 *
 *        // Input of shape (2,1,2)
 *        x = [[[ 1.1,  2.2]],
 *        [[ 3.3,  4.4]]]
 *
 *        // gamma parameter of length 1
 *        gamma = [1.5]
 *
 *        // beta parameter of length 1
 *        beta = [0.5]
 *
 *        // Instance normalization is calculated with the above formula
 *        InstanceNorm(x,gamma,beta) = [[[-0.997527  ,  1.99752665]],
 *        [[-0.99752653,  1.99752724]]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\instance_norm.cc:L80
 * \param data An n-dimensional input array (n > 2) of the form [batch, channel,
 * \param gamma A vector of length 'channel', which multiplies the normalized input.
 * \param beta A vector of length 'channel', which is added to the product of the
 * \param eps An `epsilon` parameter to prevent division by 0.
 * \return new symbol
 */
inline Symbol InstanceNorm(Symbol data,
                           Symbol gamma,
                           Symbol beta,
                           mx_float eps = 0.001) {
  return Operator("InstanceNorm")
           .SetParam("eps", eps)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol();
}

/*!
 * \breif Normalize the input array using the L2 norm.
 *
 *        For 1-D NDArray, it computes::
 *
 *        out = data / sqrt(sum(data ** 2) + eps)
 *
 *        For N-D NDArray, if the input array has shape (N, N, ..., N),
 *
 *        with ``mode`` = ``instance``, it normalizes each instance in the
 *        array by its L2 norm.::
 *
 *        for i in 0...N
 *        out[i,:,:,...,:] = data[i,:,:,...,:] / sqrt(sum(data[i,:,:,...,:] ** 2) + eps)
 *
 *        with ``mode`` = ``channel``, it normalizes each channel in the array by its L2
 *
 *        for i in 0...N
 *        out[:,i,:,...,:] = data[:,i,:,...,:] / sqrt(sum(data[:,i,:,...,:] ** 2) + eps)
 *
 *        with ``mode`` = ``spatial``, it normalizes the cross channel norm for each
 *        in the array by its L2 norm.::
 *
 *        for dim in 2...N
 *        for i in 0...N
 *        out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out,
 *        -dim-
 *
 *        Example::
 *
 *        x = [[[1,2],
 *        [3,4]],
 *        [[2,2],
 *        [5,6]]]
 *
 *        L2Normalization(x, mode='instance')
 *        =[[[ 0.18257418  0.36514837]
 *        [ 0.54772252  0.73029673]]
 *        [[ 0.24077171  0.24077171]
 *        [ 0.60192931  0.72231513]]]
 *
 *        L2Normalization(x, mode='channel')
 *        =[[[ 0.31622776  0.44721359]
 *        [ 0.94868326  0.89442718]]
 *        [[ 0.37139067  0.31622776]
 *        [ 0.92847669  0.94868326]]]
 *
 *        L2Normalization(x, mode='spatial')
 *        =[[[ 0.44721359  0.89442718]
 *        [ 0.60000002  0.80000001]]
 *        [[ 0.70710677  0.70710677]
 *        [ 0.6401844   0.76822126]]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\l2_normalization.cc:L74
 * \param data Input array to normalize.
 * \param eps A small constant for numerical stability.
 * \param mode Specify the dimension along which to compute L2 norm.
 * \return new symbol
 */
inline Symbol L2Normalization(Symbol data,
                              mx_float eps = 1e-10,
                              L2NormalizationMode mode = L2NormalizationMode::instance) {
  static const char *L2NormalizationModeValues[] = {
    "channel",
    "instance",
    "spatial"
  };
  return Operator("L2Normalization")
           .SetParam("eps", eps)
           .SetParam("mode", L2NormalizationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Applies local response normalization to the input.
 *
 *        The local response normalization layer performs "lateral inhibition" by
 *        over local input regions.
 *
 *        If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel
 *        :math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized
 *        activity :math:`b_{x,y}^{i}` is given by the expression:
 *
 *        .. math::
 *        b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \alpha \sum_{j=max(0,
 *
 *        where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial
 *        number of kernels in the layer.
 *
 *
 *
 *        Defined in
 * \param data Input data.
 * \param nsize normalization window width in elements.
 * \param alpha The variance scaling parameter :math:`lpha` in the LRN expression.
 * \param beta The power parameter :math:`eta` in the LRN expression.
 * \param knorm The parameter :math:`k` in the LRN expression.
 * \return new symbol
 */
inline Symbol LRN(Symbol data,
                  uint32_t nsize,
                  mx_float alpha = 0.0001,
                  mx_float beta = 0.75,
                  mx_float knorm = 2) {
  return Operator("LRN")
           .SetParam("nsize", nsize)
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetParam("knorm", knorm)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Make your own loss function in network construction.
 *
 *        This operator accepts a customized loss function symbol as a terminal loss and
 *        the symbol should be an operator with no backward dependency.
 *        The output of this function is the gradient of loss with respect to the input
 *
 *        For example, if you are a making a cross entropy loss function. Assume ``out``
 *        predicted output and ``label`` is the true label, then the cross entropy can be
 *
 *        cross_entropy = label * log(out) + (1 - label) * log(1 - out)
 *        loss = MakeLoss(cross_entropy)
 *
 *        We will need to use ``MakeLoss`` when we are creating our own loss function or
 *        combine multiple loss functions. Also we may want to stop some variables'
 *        from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.
 *
 *        In addition, we can give a scale to the loss by setting ``grad_scale``,
 *        so that the gradient of the loss will be rescaled in the backpropagation.
 *
 *        .. note:: This operator should be used as a Symbol instead of NDArray.
 *
 *
 *
 *        Defined in
 * \param data Input array.
 * \param grad_scale Gradient scale as a supplement to unary and binary operators
 * \param valid_thresh clip each element in the array to 0 when it is less than
 * \param normalization If this is set to null, the output gradient will not be
 *        normalized. If this is set to batch, the output gradient will be divided by the
 *        batch size. If this is set to valid, the output gradient will be divided by the
 * \return new symbol
 */
inline Symbol MakeLoss(Symbol data,
                       mx_float grad_scale = 1,
                       mx_float valid_thresh = 0,
                       MakeLossNormalization normalization = MakeLossNormalization::null) {
  static const char *MakeLossNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("MakeLoss")
           .SetParam("grad_scale", grad_scale)
           .SetParam("valid_thresh", valid_thresh)
           .SetParam("normalization", MakeLossNormalizationValues[int(normalization)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Performs pooling on the input.
 *
 *        The shapes for 1-D pooling are
 *
 *        - **data**: *(batch_size, channel, width)*,
 *        - **out**: *(batch_size, num_filter, out_width)*.
 *
 *        The shapes for 2-D pooling are
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*, with::
 *
 *        out_height = f(height, kernel[0], pad[0], stride[0])
 *        out_width = f(width, kernel[1], pad[1], stride[1])
 *
 *        The defintion of *f* depends on ``pooling_convention``, which has two options:
 *
 *        - **valid** (default)::
 *
 *        f(x, k, p, s) = floor((x+2*p-k)/s)+1
 *
 *        - **full**, which is compatible with Caffe::
 *
 *        f(x, k, p, s) = ceil((x+2*p-k)/s)+1
 *
 *        But ``global_pool`` is set to be true, then do a global pooling, namely reset
 *        ``kernel=(height, width)``.
 *
 *        Three pooling options are supported by ``pool_type``:
 *
 *        - **avg**: average pooling
 *        - **max**: max pooling
 *        - **sum**: sum pooling
 *
 *        For 3-D pooling, an additional *depth* dimension is added before
 *        *height*. Namely the input data will have shape *(batch_size, channel, depth,
 *        height, width)*.
 *
 *
 *
 *        Defined in
 * \param data Input data to the pooling operator.
 * \param kernel pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current input
 * \param cudnn_off Turn off cudnn pooling and use MXNet pooling operator.
 * \param pooling_convention Pooling convention to be applied.
 * \param stride stride: for pooling (y, x) or (d, y, x)
 * \param pad pad for pooling: (y, x) or (d, y, x)
 * \return new symbol
 */
inline Symbol Pooling(Symbol data,
                      Shape kernel,
                      PoolingPoolType pool_type,
                      bool global_pool = false,
                      bool cudnn_off = false,
                      PoolingPoolingConvention pooling_convention = PoolingPoolingConvention::valid,
                      Shape stride = Shape(),
                      Shape pad = Shape()) {
  static const char *PoolingPoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  static const char *PoolingPoolingConventionValues[] = {
    "full",
    "valid"
  };
  return Operator("Pooling")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", PoolingPoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("pooling_convention", PoolingPoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif This operator is DEPRECATED.
 *        Perform pooling on the input.
 *
 *        The shapes for 2-D pooling is
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*, with::
 *
 *        out_height = f(height, kernel[0], pad[0], stride[0])
 *        out_width = f(width, kernel[1], pad[1], stride[1])
 *
 *        The defintion of *f* depends on ``pooling_convention``, which has two options:
 *
 *        - **valid** (default)::
 *
 *        f(x, k, p, s) = floor((x+2*p-k)/s)+1
 *
 *        - **full**, which is compatible with Caffe::
 *
 *        f(x, k, p, s) = ceil((x+2*p-k)/s)+1
 *
 *        But ``global_pool`` is set to be true, then do a global pooling, namely reset
 *        ``kernel=(height, width)``.
 *
 *        Three pooling options are supported by ``pool_type``:
 *
 *        - **avg**: average pooling
 *        - **max**: max pooling
 *        - **sum**: sum pooling
 *
 *        1-D pooling is special case of 2-D pooling with *weight=1* and
 *        *kernel[1]=1*.
 *
 *        For 3-D pooling, an additional *depth* dimension is added before
 *        *height*. Namely the input data will have shape *(batch_size, channel, depth,
 *        height, width)*.
 *
 *
 *
 *        Defined in
 * \param data Input data to the pooling operator.
 * \param kernel pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current input
 * \param pooling_convention Pooling convention to be applied.
 * \param stride stride: for pooling (y, x) or (d, y, x)
 * \param pad pad for pooling: (y, x) or (d, y, x)
 * \return new symbol
 */
inline Symbol Pooling_v1(Symbol data,
                         Shape kernel,
                         Pooling_v1PoolType pool_type,
                         bool global_pool = false,
                         Pooling_v1PoolingConvention pooling_convention = Pooling_v1PoolingConvention::valid,
                         Shape stride = Shape(),
                         Shape pad = Shape()) {
  static const char *Pooling_v1PoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  static const char *Pooling_v1PoolingConventionValues[] = {
    "full",
    "valid"
  };
  return Operator("Pooling_v1")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", Pooling_v1PoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("pooling_convention", Pooling_v1PoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes and optimizes for squared loss.
 *
 *        .. note::
 *        Use the LinearRegressionOutput as the final output layer of a net.
 *
 *        By default, gradients of this loss function are scaled by factor `1/n`, where n
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\regression_output.cc:L45
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LinearRegressionOutput(Symbol data,
                                     Symbol label,
                                     mx_float grad_scale = 1) {
  return Operator("LinearRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Computes mean absolute error of the input.
 *
 *        MAE is a risk metric corresponding to the expected value of the absolute error.
 *
 *        If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i`
 *        then the mean absolute error (MAE) estimated over :math:`n` samples is defined
 *
 *        :math:`\text{MAE}(y, \hat{y} ) = \frac{1}{n} \sum_{i=0}^{n-1} \left| y_i -
 *
 *        .. note::
 *        Use the MAERegressionOutput as the final output layer of a net.
 *
 *        By default, gradients of this loss function are scaled by factor `1/n`, where n
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\regression_output.cc:L66
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol MAERegressionOutput(Symbol data,
                                  Symbol label,
                                  mx_float grad_scale = 1) {
  return Operator("MAERegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Applies a logistic function to the input.
 *
 *        The logistic function, also known as the sigmoid function, is computed as
 *        :math:`\frac{1}{1+exp(-x)}`.
 *
 *        Commonly, the sigmoid is used to squash the real-valued output of a linear model
 *        :math:wTx+b into the [0,1] range so that it can be interpreted as a probability.
 *        It is suitable for binary classification or probability prediction tasks.
 *
 *        .. note::
 *        Use the LogisticRegressionOutput as the final output layer of a net.
 *
 *        By default, gradients of this loss function are scaled by factor `1/n`, where n
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\regression_output.cc:L87
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LogisticRegressionOutput(Symbol data,
                                       Symbol label,
                                       mx_float grad_scale = 1) {
  return Operator("LogisticRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Applies a recurrent layer to input.
 * \param data Input data to RNN
 * \param parameters Vector of all RNN trainable parameters concatenated
 * \param state initial hidden state of the RNN
 * \param state_cell initial cell state for LSTM networks (only for LSTM)
 * \param state_size size of the state for each layer
 * \param num_layers number of stacked layers
 * \param mode the type of RNN to compute
 * \param bidirectional whether to use bidirectional recurrent layers
 * \param p Dropout probability, fraction of the input that gets dropped out at training
 * \param state_outputs Whether to have the states as symbol outputs.
 * \return new symbol
 */
inline Symbol RNN(Symbol data,
                  Symbol parameters,
                  Symbol state,
                  Symbol state_cell,
                  uint32_t state_size,
                  uint32_t num_layers,
                  RNNMode mode,
                  bool bidirectional = false,
                  mx_float p = 0,
                  bool state_outputs = false) {
  static const char *RNNModeValues[] = {
    "gru",
    "lstm",
    "rnn_relu",
    "rnn_tanh"
  };
  return Operator("RNN")
           .SetParam("state_size", state_size)
           .SetParam("num_layers", num_layers)
           .SetParam("mode", RNNModeValues[int(mode)])
           .SetParam("bidirectional", bidirectional)
           .SetParam("p", p)
           .SetParam("state_outputs", state_outputs)
           .SetInput("data", data)
           .SetInput("parameters", parameters)
           .SetInput("state", state)
           .SetInput("state_cell", state_cell)
           .CreateSymbol();
}

/*!
 * \breif Performs region of interest(ROI) pooling on the input array.
 *
 *        ROI pooling is a variant of a max pooling layer, in which the output size is
 *        region of interest is a parameter. Its purpose is to perform max pooling on the
 *        of non-uniform sizes to obtain fixed-size feature maps. ROI pooling is a
 *        layer mostly used in training a `Fast R-CNN` network for object detection.
 *
 *        This operator takes a 4D feature map as an input array and region proposals as
 *        then it pools over sub-regions of input and produces a fixed-sized output array
 *        regardless of the ROI size.
 *
 *        To crop the feature map accordingly, you can resize the bounding box coordinates
 *        by changing the parameters `rois` and `spatial_scale`.
 *
 *        The cropped feature maps are pooled by standard max pooling operation to a
 *        indicated by a `pooled_size` parameter. batch_size will change to the number of
 *        bounding boxes after `ROIPooling`.
 *
 *        The size of each region of interest doesn't have to be perfectly divisible by
 *        the number of pooling sections(`pooled_size`).
 *
 *        Example::
 *
 *        x = [[[[  0.,   1.,   2.,   3.,   4.,   5.],
 *        [  6.,   7.,   8.,   9.,  10.,  11.],
 *        [ 12.,  13.,  14.,  15.,  16.,  17.],
 *        [ 18.,  19.,  20.,  21.,  22.,  23.],
 *        [ 24.,  25.,  26.,  27.,  28.,  29.],
 *        [ 30.,  31.,  32.,  33.,  34.,  35.],
 *        [ 36.,  37.,  38.,  39.,  40.,  41.],
 *        [ 42.,  43.,  44.,  45.,  46.,  47.]]]]
 *
 *        // region of interest i.e. bounding box coordinates.
 *        y = [[0,0,0,4,4]]
 *
 *        // returns array of shape (2,2) according to the given roi with max pooling.
 *        ROIPooling(x, y, (2,2), 1.0) = [[[[ 14.,  16.],
 *        [ 26.,  28.]]]]
 *
 *        // region of interest is changed due to the change in `spacial_scale` parameter.
 *        ROIPooling(x, y, (2,2), 0.7) = [[[[  7.,   9.],
 *        [ 19.,  21.]]]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\roi_pooling.cc:L273
 * \param data The input array to the pooling operator,  a 4D Feature maps
 * \param rois Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]],
 *        where (x1, y1) and (x2, y2) are top left and bottom right corners of designated
 *        region of interest. `batch_index` indicates the index of corresponding image in
 * \param pooled_size ROI pooling output shape (h,w)
 * \param spatial_scale Ratio of input feature map height (or w) to raw image height (or
 * \return new symbol
 */
inline Symbol ROIPooling(Symbol data,
                         Symbol rois,
                         Shape pooled_size,
                         mx_float spatial_scale) {
  return Operator("ROIPooling")
           .SetParam("pooled_size", pooled_size)
           .SetParam("spatial_scale", spatial_scale)
           .SetInput("data", data)
           .SetInput("rois", rois)
           .CreateSymbol();
}

/*!
 * \breif Takes the last element of a sequence.
 *
 *        This function takes an n-dimensional input array of the form
 *        [max_sequence_length, batch_size, other_feature_dims] and returns a
 *        of the form [batch_size, other_feature_dims].
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        an input array of positive ints of dimension [batch_size]. To use this
 *        set `use_sequence_length` to `True`, otherwise each example in the batch is
 *        to have the max sequence length.
 *
 *        .. note:: Alternatively, you can also use `take` operator.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.],
 *        [  7.,   8.,   9.]],
 *
 *        [[ 10.,   11.,   12.],
 *        [ 13.,   14.,   15.],
 *        [ 16.,   17.,   18.]],
 *
 *        [[  19.,   20.,   21.],
 *        [  22.,   23.,   24.],
 *        [  25.,   26.,   27.]]]
 *
 *        // returns last sequence when sequence_length parameter is not used
 *        SequenceLast(x) = [[  19.,   20.,   21.],
 *        [  22.,   23.,   24.],
 *        [  25.,   26.,   27.]]
 *
 *        // sequence_length y is used
 *        SequenceLast(x, y=[1,1,1], use_sequence_length=True) =
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.],
 *        [  7.,   8.,   9.]]
 *
 *        // sequence_length y is used
 *        SequenceLast(x, y=[1,2,3], use_sequence_length=True) =
 *        [[  1.,    2.,   3.],
 *        [  13.,  14.,  15.],
 *        [  25.,  26.,  27.]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\sequence_last.cc:L77
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \return new symbol
 */
inline Symbol SequenceLast(Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false) {
  return Operator("SequenceLast")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \breif Sets all elements outside the sequence to a constant value.
 *
 *        This function takes an n-dimensional input array of the form
 *        [max_sequence_length, batch_size, other_feature_dims] and returns an array of
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        should be an input array of positive ints of dimension [batch_size].
 *        To use this parameter, set `use_sequence_length` to `True`,
 *        otherwise each example in the batch is assumed to have the max sequence length
 *        this operator works as the `identity` operator.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // Batch 1
 *        B1 = [[  1.,   2.,   3.],
 *        [  7.,   8.,   9.],
 *        [ 13.,  14.,  15.]]
 *
 *        // Batch 2
 *        B2 = [[  4.,   5.,   6.],
 *        [ 10.,  11.,  12.],
 *        [ 16.,  17.,  18.]]
 *
 *        // works as identity operator when sequence_length parameter is not used
 *        SequenceMask(x) = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // sequence_length [1,1] means 1 of each batch will be kept
 *        // and other rows are masked with default mask value = 0
 *        SequenceMask(x, y=[1,1], use_sequence_length=True) =
 *        [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  0.,   0.,   0.],
 *        [  0.,   0.,   0.]],
 *
 *        [[  0.,   0.,   0.],
 *        [  0.,   0.,   0.]]]
 *
 *        // sequence_length [2,3] means 2 of batch B1 and 3 of batch B2 will be kept
 *        // and other rows are masked with value = 1
 *        SequenceMask(x, y=[2,3], use_sequence_length=True, value=1) =
 *        [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [  10.,  11.,  12.]],
 *
 *        [[   1.,   1.,   1.],
 *        [  16.,  17.,  18.]]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\sequence_mask.cc:L112
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \param value The value to be used as a mask.
 * \return new symbol
 */
inline Symbol SequenceMask(Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false,
                           mx_float value = 0) {
  return Operator("SequenceMask")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetParam("value", value)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \breif Reverses the elements of each sequence.
 *
 *        This function takes an n-dimensional input array of the form
 *        and returns an array of the same shape.
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        `sequence_length` should be an input array of positive ints of dimension
 *        To use this parameter, set `use_sequence_length` to `True`,
 *        otherwise each example in the batch is assumed to have the max sequence length.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // Batch 1
 *        B1 = [[  1.,   2.,   3.],
 *        [  7.,   8.,   9.],
 *        [ 13.,  14.,  15.]]
 *
 *        // Batch 2
 *        B2 = [[  4.,   5.,   6.],
 *        [ 10.,  11.,  12.],
 *        [ 16.,  17.,  18.]]
 *
 *        // returns reverse sequence when sequence_length parameter is not used
 *        SequenceReverse(x) = [[[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]]]
 *
 *        // sequence_length [2,2] means 2 rows of
 *        // both batch B1 and B2 will be reversed.
 *        SequenceReverse(x, y=[2,2], use_sequence_length=True) =
 *        [[[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // sequence_length [2,3] means 2 of batch B2 and 3 of batch B3
 *        // will be reversed.
 *        SequenceReverse(x, y=[2,3], use_sequence_length=True) =
 *        [[[  7.,   8.,   9.],
 *        [ 16.,  17.,  18.]],
 *
 *        [[  1.,   2.,   3.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14,   15.],
 *        [  4.,   5.,   6.]]]
 *
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\sequence_reverse.cc:L98
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \return new symbol
 */
inline Symbol SequenceReverse(Symbol data,
                              Symbol sequence_length,
                              bool use_sequence_length = false) {
  return Operator("SequenceReverse")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \breif Applies softmax activation to input. This is intended for internal layers. For
 *        output (loss layer) please use SoftmaxOutput. If mode=instance, this operator
 *        will compute a softmax for each instance in the batch; this is the default
 *        mode. If mode=channel, this operator will compute a num_channel-class softmax
 *        at each position of each instance; this can be used for fully convolutional
 * \param data Input data to activation function.
 * \param mode Softmax Mode. If set to instance, this operator will compute a softmax for
 *        each instance in the batch; this is the default mode. If set to channel, this
 *        operator will compute a num_channel-class softmax at each position of each
 *        instance; this can be used for fully convolutional network, image segmentation,
 * \return new symbol
 */
inline Symbol SoftmaxActivation(Symbol data,
                                SoftmaxActivationMode mode = SoftmaxActivationMode::instance) {
  static const char *SoftmaxActivationModeValues[] = {
    "channel",
    "instance"
  };
  return Operator("SoftmaxActivation")
           .SetParam("mode", SoftmaxActivationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Computes softmax with logit loss.
 *
 *        In the forward pass, the softmax output is returned. Assume the input data has
 *        shape *(n,k)*, then the output will have the same shape as the input, which is
 *
 *        .. math::
 *        out[i,:] = softmax(data[i,:])
 *
 *        for :math:`i=0,...,n-1`, where
 *
 *        .. math::
 *        softmax(x) = \left[..., \frac{exp(x[j])}{exp(x[0])+...+exp(x[k-1])}, ...\right]
 *
 *        For general *N*-D input array with shape :math:`(d_1, ..., d_n)`. Denoted by
 *        :math:`s=d_1d_2...d_n`. The way to compute softmax various:
 *
 *        - ``preserve_shape`` is false (default). Reshape input into a 2-D array with
 *        shape :math:`(d_1, s/d_1)` beforing computing the softmax, and then reshaped
 *        original shape.
 *
 *        - ``preserve_shape`` is true. For all :math:`i_1, ..., i_{n-1}`, compute
 *
 *        .. math::
 *        out[i_1, ..., i_{n-1}, :] = softmax(data[i_1, ..., i_{n-1},:])
 *
 *        - ``multi_output`` is true. For all :math:`i_1, ..., i_{n-1}`, compute
 *
 *        .. math::
 *        out[i_1, :, ..., i_{n-1}] = softmax(data[i_1, :, ..., i_{n-1}])
 *
 *        In the backward pass, the logit loss, also called cross-entroy loss, is
 *        added. The provided label can be a *(N-1)*-D label index array or a *N*-D label
 *        probability array.
 *
 *        Examples with a particular label can be ignored during backward by specifying
 *        ``ignore_label`` (also need ``use_ignore`` to be true).
 *
 *        A scale can be applied to the gradient by ``grad_scale``, which is often used in
 *        mutli-loss object function in which we can given each loss different weight. It
 *        also supports various ways to normalize the gradient by ``normalization``:
 *
 *        - **null**: do nothing
 *        - **batch**: divide by batch size (number of examples)
 *        - **valid**: divide by the number of examples which are not ignored.
 *
 *
 *        Defined in
 *        E:\CI-Cor-Ready\ai\face-demo\app\externals\mxnet\src\operator\softmax_output.cc:L77
 * \param data Input data.
 * \param label Ground truth label.
 * \param grad_scale Scale the gradient by a float factor
 * \param ignore_label the labels with value equals to ``ignore_label`` will be ignored
 * \param multi_output If set to true, softmax will applied on axis 1
 * \param use_ignore If set to true, the ignore_label value will not contribute to the
 * \param preserve_shape If true, softmax will applied on the last axis
 * \param normalization Normalize the gradient
 * \param out_grad Apply weighting from output gradient
 * \return new symbol
 */
inline Symbol SoftmaxOutput(Symbol data,
                            Symbol label,
                            mx_float grad_scale = 1,
                            mx_float ignore_label = -1,
                            bool multi_output = false,
                            bool use_ignore = false,
                            bool preserve_shape = false,
                            SoftmaxOutputNormalization normalization = SoftmaxOutputNormalization::null,
                            bool out_grad = false) {
  static const char *SoftmaxOutputNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("SoftmaxOutput")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxOutputNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Perform a softmax transformation on input. Please use SoftmaxOutput.. note::
 * \param data Input data to softmax.
 * \param grad_scale Scale the gradient by a float factor
 * \param ignore_label the labels with value equals to ``ignore_label`` will be ignored
 * \param multi_output If set to true, softmax will applied on axis 1
 * \param use_ignore If set to true, the ignore_label value will not contribute to the
 * \param preserve_shape If true, softmax will applied on the last axis
 * \param normalization Normalize the gradient
 * \param out_grad Apply weighting from output gradient
 * \return new symbol
 */
inline Symbol Softmax(Symbol data,
                      mx_float grad_scale = 1,
                      mx_float ignore_label = -1,
                      bool multi_output = false,
                      bool use_ignore = false,
                      bool preserve_shape = false,
                      SoftmaxNormalization normalization = SoftmaxNormalization::null,
                      bool out_grad = false) {
  static const char *SoftmaxNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("Softmax")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \breif Applies a spatial transformer to input feature map.
 * \param data Input data to the SpatialTransformerOp.
 * \param loc localisation net, the output dim should be 6 when transform_type is affine.
 * \param transform_type transformation type
 * \param sampler_type sampling type
 * \param target_shape output shape(h, w) of spatial transformer: (y, x)
 * \return new symbol
 */
inline Symbol SpatialTransformer(Symbol data,
                                 Symbol loc,
                                 SpatialTransformerTransformType transform_type,
                                 SpatialTransformerSamplerType sampler_type,
                                 Shape target_shape = Shape(0,0)) {
  static const char *SpatialTransformerTransformTypeValues[] = {
    "affine"
  };
  static const char *SpatialTransformerSamplerTypeValues[] = {
    "bilinear"
  };
  return Operator("SpatialTransformer")
           .SetParam("transform_type", SpatialTransformerTransformTypeValues[int(transform_type)])
           .SetParam("sampler_type", SpatialTransformerSamplerTypeValues[int(sampler_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .SetInput("loc", loc)
           .CreateSymbol();
}

/*!
 * \breif Computes support vector machine based transformation of the input.
 *
 *        This tutorial demonstrates using SVM as output layer for classification instead
 *        https://github.com/dmlc/mxnet/tree/master/example/svm_mnist.
 *
 *
 * \param data Input data for SVM transformation.
 * \param label Class label for the input data.
 * \param margin The loss function penalizes outputs that lie outside this margin.
 * \param regularization_coefficient Regularization parameter for the SVM. This balances
 * \param use_linear Whether to use L1-SVM objective. L2-SVM objective is used by default.
 * \return new symbol
 */
inline Symbol SVMOutput(Symbol data,
                        Symbol label,
                        mx_float margin = 1,
                        mx_float regularization_coefficient = 1,
                        bool use_linear = false) {
  return Operator("SVMOutput")
           .SetParam("margin", margin)
           .SetParam("regularization_coefficient", regularization_coefficient)
           .SetParam("use_linear", use_linear)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \breif Apply a custom operator implemented in a frontend language (like Python).
 *
 *        Custom operators should override required methods like `forward` and `backward`.
 *        The custom operator must be registered before it can be used.
 *        Please check the tutorial here: http://mxnet.io/how_to/new_op.html.
 *
 *
 * \param op_type Name of the custom operator. This is the name that is passed to
 * \param data Input data for the custom operator.
 * \return new symbol
 */
inline Symbol Custom(const std::string& op_type,
                     Symbol data) {
  return Operator("Custom")
           .SetInput("data", data)
           .CreateSymbol();
}

} //namespace cpp
} //namespace mxnet
#endif  // CPP_PACKAGE_INCLUDE_MXNET_CPP_OP_H_
