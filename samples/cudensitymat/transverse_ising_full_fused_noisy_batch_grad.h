/* Copyright (c) 2026-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cudensitymat.h> // cuDensityMat library header
#include "helpers.h"      // GPU helper functions

#include <cmath>
#include <complex>
#include <vector>
#include <iostream>
#include <cassert>


/* DESCRIPTION:
   Batched time-dependent transverse-field Ising Hamiltonian operator
   with ordered and fused ZZ terms, plus fused unitary dissipation terms:
    H[k] = sum_{i} {h_i(t)[k] * X_i}          // transverse-field sum of X_i operators with batched time-dependent h_i(t)[k] coefficients 
      + f(t)[k] * sum_{i < j} {g_ij * ZZ_ij}  // batched modulated sum of the fused ordered {Z_i * Z_j} terms with static g_ij coefficients
      + d * sum_{i} {Y_i * {..} * Y_i}        // scaled sum of the dissipation terms {Y_i * {..} * Y_i} fused into the YY_ii super-operators
   where {..} is the placeholder for the density matrix to show that the Y_i operators act from different sides.
*/

/** Define the numerical type and data type for the GPU computations (same) */
using NumericalType = std::complex<double>;      // do not change
constexpr cudaDataType_t dataType = CUDA_C_64F;  // do not change


/** User-provided batched scalar CPU callback C function
 *  defining a batched time-dependent coefficient h_i(t) for all instances
 *  of the batch inside the Hamiltonian:
 *  h_i(t)[k] = exp(-Omega[k] * t) for k = 0, ..., batchSize-1
 */
extern "C"
int32_t hCoefBatchComplex64(
  double time,             //in: time point
  int64_t batchSize,       //in: user-defined batch size (number of coefficients in the batch)
  int32_t numParams,       //in: number of external user-provided Hamiltonian parameters (this function expects one parameter, Omega)
  const double * params,   //in: params[0:numParams-1][0:batchSize-1]: GPU-accessible F-ordered array of user-provided Hamiltonian parameters for all instances of the batch
  cudaDataType_t dataType, //in: data type (expecting CUDA_C_64F in this specific callback function)
  void * scalarStorage,    //inout: CPU-accessible storage for the returned batched coefficient values of shape [0:batchSize-1]
  cudaStream_t stream)     //in: CUDA stream (default is 0x0)
{
  if (dataType == CUDA_C_64F) {
    auto * tdCoef = static_cast<cuDoubleComplex *>(scalarStorage); // casting to cuDoubleComplex because this callback function expects CUDA_C_64F data type
    for (int64_t k = 0; k < batchSize; ++k) { // for each instance of the batch
      const auto omega = params[k * numParams + 0]; // params[0][k]: 0-th parameter for k-th instance of the batch
      tdCoef[k] = make_cuDoubleComplex(std::exp((-omega) * time), 0.0); // value of the k-th instance of the batched coefficients
    }
  } else {
    return 1; // error code (1: Error)
  }
  return 0; // error code (0: Success)
}


/** User-provided batched scalar gradient callback function (CPU-side) for the user-provided
 *  batched scalar callback function hCoefBatchComplex64, defining the gradients with respect
 *  to its single (batched) parameter Omega. It accumulates a partial derivative:
 *    2*Re(dCost/dOmega[k]) = 2*Re(dCost/dCoef[k] * dCoef[k]/dOmega[k]),
 *  where:
 *  - Cost is some user-defined real scalar cost function,
 *  - dCost/dCoef[k] is the adjoint of the cost function with respect to the k-th instance of the batched coefficient associated with the callback function,
 *  - dCoef[k]/dOmega[k] is the gradient of the k-th instance of the batched coefficient with respect to the parameter Omega[k]:
 *    dCoef[k]/dOmega[k] = -t * exp(-Omega[k] * t) for k = 0, ..., batchSize-1
 */
extern "C"
int32_t hCoefBatchGradComplex64(
  double time,             //in: time point
  int64_t batchSize,       //in: user-defined batch size (number of coefficients in the batch)
  int32_t numParams,       //in: number of external user-provided Hamiltonian parameters (this function expects one parameter, Omega)
  const double * params,   //in: params[0:numParams-1][0:batchSize-1]: GPU-accessible F-ordered array of user-provided Hamiltonian parameters for all instances of the batch
  cudaDataType_t dataType, //in: data type (expecting CUDA_C_64F in this specific callback function)
  void * scalarGrad,       //in: CPU-accessible storage for the batched adjoint of the batched coefficient of shape [0:batchSize-1]
  double * paramsGrad,     //inout: params[0:numParams-1][0:batchSize-1]: GPU-accessible F-ordered array of the returned gradients of the parameter(s) for all instances of the batch
  cudaStream_t stream)     //in: CUDA stream (default is 0x0)
{
  if (dataType == CUDA_C_64F) {
    const auto * tdCoefAdjoint = static_cast<const cuDoubleComplex *>(scalarGrad); // casting to cuDoubleComplex because this callback function expects CUDA_C_64F data type
    for (int64_t k = 0; k < batchSize; ++k) { // for each instance of the batch
      const auto omega = params[k * numParams + 0]; // params[0][k]: 0-th parameter for k-th instance of the batch
      paramsGrad[k * numParams + 0] += // IMPORTANT: Accumulate the partial derivative for the k-th instance of the batch, not overwrite it!
        2.0 * cuCreal(cuCmul(tdCoefAdjoint[k], make_cuDoubleComplex(std::exp((-omega) * time) * (-time), 0.0)));
    }
  } else {
    return 1; // error code (1: Error)
  }
  return 0; // error code (0: Success)
}


/** User-provided batched scalar CPU callback C function
 *  defining a batched time-dependent coefficient f(t) for all instances
 *  of the batch inside the Hamiltonian:
 *  f(t)[k] = exp(i * Omega[k] * t)
 *          = cos(Omega[k] * t) + i * sin(Omega[k] * t) for k = 0, ..., batchSize-1
 */
extern "C"
int32_t fCoefBatchComplex64(
  double time,             //in: time point
  int64_t batchSize,       //in: user-defined batch size (number of coefficients in the batch)
  int32_t numParams,       //in: number of external user-provided Hamiltonian parameters (this function expects one parameter, Omega)
  const double * params,   //in: params[0:numParams-1][0:batchSize-1]: GPU-accessible F-ordered array of user-provided Hamiltonian parameters for all instances of the batch
  cudaDataType_t dataType, //in: data type (expecting CUDA_C_64F in this specific callback function)
  void * scalarStorage,    //inout: CPU-accessible storage for the returned batched coefficient values of shape [0:batchSize-1]
  cudaStream_t stream)     //in: CUDA stream (default is 0x0)
{
  if (dataType == CUDA_C_64F) {
    auto * tdCoef = static_cast<cuDoubleComplex *>(scalarStorage); // casting to cuDoubleComplex because this callback function expects CUDA_C_64F data type
    for (int64_t k = 0; k < batchSize; ++k) { // for each instance of the batch
      const auto omega = params[k * numParams + 0]; // params[0][k]: 0-th parameter for k-th instance of the batch
      tdCoef[k] = make_cuDoubleComplex(std::cos(omega * time), std::sin(omega * time)); // value of the k-th instance of the batched coefficients
    }
  } else {
    return 1; // error code (1: Error)
  }
  return 0; // error code (0: Success)
}


/** User-provided batched scalar gradient callback function (CPU-side) for the user-provided
 *  batched scalar callback function fCoefBatchComplex64, defining the gradients with respect
 *  to its single (batched) parameter Omega. It accumulates a partial derivative:
 *    2*Re(dCost/dOmega[k]) = 2*Re(dCost/dCoef[k] * dCoef[k]/dOmega[k]),
 *  where:
 *  - Cost is some user-defined real scalar cost function,
 *  - dCost/dCoef[k] is the adjoint of the cost function with respect to the k-th instance of the batched coefficient associated with the callback function,
 *  - dCoef[k]/dOmega[k] is the gradient of the k-th instance of the batched coefficient with respect to the parameter Omega[k]:
 *    dCoef[k]/dOmega[k] = i * t * exp(i * Omega[k] * t)
 *                       = i * t * cos(Omega[k] * t) - t * sin(Omega[k] * t) for k = 0, ..., batchSize-1
 */
extern "C"
int32_t fCoefBatchGradComplex64(
  double time,             //in: time point
  int64_t batchSize,       //in: user-defined batch size (number of coefficients in the batch)
  int32_t numParams,       //in: number of external user-provided Hamiltonian parameters (this function expects one parameter, Omega)
  const double * params,   //in: params[0:numParams-1][0:batchSize-1]: GPU-accessible F-ordered array of user-provided Hamiltonian parameters for all instances of the batch
  cudaDataType_t dataType, //in: data type (expecting CUDA_C_64F in this specific callback function)
  void * scalarGrad,       //in: CPU-accessible storage for the batched adjoint of the batched coefficient of shape [0:batchSize-1]
  double * paramsGrad,     //inout: params[0:numParams-1][0:batchSize-1]: GPU-accessible F-ordered array of the returned gradients of the parameter(s) for all instances of the batch
  cudaStream_t stream)     //in: CUDA stream (default is 0x0)
{
  if (dataType == CUDA_C_64F) {
    const auto * tdCoefAdjoint = static_cast<const cuDoubleComplex *>(scalarGrad); // casting to cuDoubleComplex because this callback function expects CUDA_C_64F data type
    for (int64_t k = 0; k < batchSize; ++k) {
      const auto omega = params[k * numParams + 0]; // params[0][k]: 0-th parameter for k-th instance of the batch
      paramsGrad[k * numParams + 0] += // IMPORTANT: Accumulate the partial derivative for the k-th instance of the batch, not overwrite it!
        2.0 * cuCreal(cuCmul(tdCoefAdjoint[k], make_cuDoubleComplex(-std::sin(omega * time) * time, std::cos(omega * time) * time)));
    }
  } else {
    return 1; // error code (1: Error)
  }
  return 0; // error code (0: Success)
}


/** Convenience class which encapsulates a user-defined Liouvillian operator (system Hamiltonian + dissipation terms):
 *  - Constructor constructs the desired Liouvillian operator (`cudensitymatOperator_t`)
 *  - Method `get()` returns a reference to the constructed Liouvillian operator
 *  - Destructor releases all resources used by the Liouvillian operator
 */
class UserDefinedLiouvillian final
{
private:
  // Data members
  cudensitymatHandle_t handle;             // library context handle
  int64_t operBatchSize;                   // batch size for the super-operator
  const std::vector<int64_t> spaceShape;   // Hilbert space shape (extents of the modes of the composite Hilbert space)
  void * spinXelems {nullptr};             // elements of the X spin operator in GPU RAM (F-order storage)
  void * spinYYelems {nullptr};            // elements of the fused YY two-spin operator in GPU RAM (F-order storage)
  void * spinZZelems {nullptr};            // elements of the fused ZZ two-spin operator in GPU RAM (F-order storage)
  cudensitymatElementaryOperator_t spinX;  // X spin operator (elementary tensor operator)
  cudensitymatElementaryOperator_t spinYY; // fused YY two-spin operator (elementary tensor operator)
  cudensitymatElementaryOperator_t spinZZ; // fused ZZ two-spin operator (elementary tensor operator)
  cudensitymatOperatorTerm_t oneBodyTerm;  // operator term: H1 = sum_{i} {h_i(t) * X_i} (one-body term)
  cudensitymatOperatorTerm_t twoBodyTerm;  // operator term: H2 = f(t) * sum_{i < j} {g_ij * ZZ_ij} (two-body term)
  cudensitymatOperatorTerm_t noiseTerm;    // operator term: D1 = d * sum_{i} {YY_ii}  // Y_i operators act from different sides on the density matrix (two-body mixed term)
  // Batched coefficients
  cuDoubleComplex * hCoefsStatic {nullptr}; // static part of the h(t) batched coefficients in the one-body term (of length operBatchSize)
  cuDoubleComplex * hCoefsTotal {nullptr};  // total h(t) batched coefficients in the one-body term (of length operBatchSize)
  cuDoubleComplex * fCoefsStaticMinus {nullptr}; // static part of the f(t) batched coefficients in the two-body term (of length operBatchSize)
  cuDoubleComplex * fCoefsTotalMinus {nullptr};  // total f(t) batched coefficients in the two-body term (of length operBatchSize)
  cuDoubleComplex * fCoefsStaticPlus {nullptr};  // static part of the f(t) batched coefficients in the dual two-body term (of length operBatchSize)
  cuDoubleComplex * fCoefsTotalPlus {nullptr};   // total f(t) batched coefficients in the dual two-body term (of length operBatchSize)
  // Final Liouvillian operator
  cudensitymatOperator_t liouvillian; // full operator: (-i * (H1 + H2) * {..}) + (i * {..} * (H1 + H2)) + D1{..} (super-operator)

public:

  // Constructor constructs a user-defined Liouvillian operator
  UserDefinedLiouvillian(cudensitymatHandle_t contextHandle,             // library context handle
                         const std::vector<int64_t> & hilbertSpaceShape, // Hilbert space shape
                         int64_t batchSize):                             // batch size for the super-operator
    handle(contextHandle), operBatchSize(batchSize), spaceShape(hilbertSpaceShape)
  {
    // Define the necessary operator tensors in GPU memory (F-order storage!)
    spinXelems = createInitializeArrayGPU<NumericalType>(  // X[i0; j0]
                  {{0.0, 0.0}, {1.0, 0.0},   // 1st column of matrix X
                   {1.0, 0.0}, {0.0, 0.0}}); // 2nd column of matrix X

    spinYYelems = createInitializeArrayGPU<NumericalType>(  // YY[i0, i1; j0, j1] := Y[i0; j0] * Y[i1; j1]
                    {{0.0, 0.0},  {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0},  // 1st column of matrix YY
                     {0.0, 0.0},  {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},   // 2nd column of matrix YY
                     {0.0, 0.0},  {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},   // 3rd column of matrix YY
                     {-1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}); // 4th column of matrix YY

    spinZZelems = createInitializeArrayGPU<NumericalType>(  // ZZ[i0, i1; j0, j1] := Z[i0; j0] * Z[i1; j1]
                    {{1.0, 0.0}, {0.0, 0.0},  {0.0, 0.0},  {0.0, 0.0},   // 1st column of matrix ZZ
                     {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0},  {0.0, 0.0},   // 2nd column of matrix ZZ
                     {0.0, 0.0}, {0.0, 0.0},  {-1.0, 0.0}, {0.0, 0.0},   // 3rd column of matrix ZZ
                     {0.0, 0.0}, {0.0, 0.0},  {0.0, 0.0},  {1.0, 0.0}}); // 4th column of matrix ZZ

    // Construct the necessary Elementary Tensor Operators
    //  X_i operator
    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(handle,
                        1,                                   // one-body operator
                        std::vector<int64_t>({2}).data(),    // acts in tensor space of shape {2}
                        CUDENSITYMAT_OPERATOR_SPARSITY_NONE, // dense tensor storage
                        0,                                   // 0 for dense tensors
                        nullptr,                             // nullptr for dense tensors
                        dataType,                            // data type
                        spinXelems,                          // tensor elements in GPU memory
                        cudensitymatTensorCallbackNone,      // no tensor callback function (tensor is not time-dependent)
                        cudensitymatTensorGradientCallbackNone, // no tensor gradient callback function
                        &spinX));                            // the created elementary tensor operator
    //  ZZ_ij = Z_i * Z_j fused operator
    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(handle,
                        2,                                   // two-body operator
                        std::vector<int64_t>({2,2}).data(),  // acts in tensor space of shape {2,2}
                        CUDENSITYMAT_OPERATOR_SPARSITY_NONE, // dense tensor storage
                        0,                                   // 0 for dense tensors
                        nullptr,                             // nullptr for dense tensors
                        dataType,                            // data type
                        spinZZelems,                         // tensor elements in GPU memory
                        cudensitymatTensorCallbackNone,      // no tensor callback function (tensor is not time-dependent)
                        cudensitymatTensorGradientCallbackNone, // no tensor gradient callback function
                        &spinZZ));                           // the created elementary tensor operator
    //  YY_ii = Y_i * {..} * Y_i fused operator (note action from different sides)
    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(handle,
                        2,                                   // two-body operator
                        std::vector<int64_t>({2,2}).data(),  // acts in tensor space of shape {2,2}
                        CUDENSITYMAT_OPERATOR_SPARSITY_NONE, // dense tensor storage
                        0,                                   // 0 for dense tensors
                        nullptr,                             // nullptr for dense tensors
                        dataType,                            // data type
                        spinYYelems,                         // tensor elements in GPU memory
                        cudensitymatTensorCallbackNone,      // no tensor callback function (tensor is not time-dependent)
                        cudensitymatTensorGradientCallbackNone, // no tensor gradient callback function
                        &spinYY));                           // the created elementary tensor operator

    // Construct the necessary Operator Terms from tensor products of Elementary Tensor Operators
    //  Create an empty operator term
    HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle,
                        spaceShape.size(),                   // Hilbert space rank (number of modes)
                        spaceShape.data(),                   // Hilbert space shape (mode extents)
                        &oneBodyTerm));                      // the created empty operator term
    //  Define the batched operator term: H1[k] = sum_{i} {h_i(t)[k] * X_i}
    hCoefsStatic = static_cast<cuDoubleComplex *>(createInitializeArrayGPU(
      std::vector<cuDoubleComplex>(operBatchSize, make_cuDoubleComplex(1.0, 0.0)))); // 1.0 constant for all coefficient instances in the batch
    hCoefsTotal = static_cast<cuDoubleComplex *>(createInitializeArrayGPU(
      std::vector<cuDoubleComplex>(operBatchSize, make_cuDoubleComplex(0.0, 0.0)))); // storage for the total coefficients for all instances of the batch
    for (int32_t i = 0; i < spaceShape.size(); ++i) {
      HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProductBatch(handle,
                          oneBodyTerm,
                          1,                                                             // number of elementary tensor operators in the product
                          std::vector<cudensitymatElementaryOperator_t>({spinX}).data(), // elementary tensor operators forming the product
                          std::vector<int32_t>({i}).data(),                              // space modes acted on by the operator product
                          std::vector<int32_t>({0}).data(),                              // space mode action duality (0: from the left; 1: from the right)
                          operBatchSize,                                                 // batch size
                          hCoefsStatic,                                                  // static part of the h(t) batched coefficients in the one-body term
                          hCoefsTotal,                                                   // total h(t) batched coefficients in the one-body term
                          {hCoefBatchComplex64, CUDENSITYMAT_CALLBACK_DEVICE_CPU, nullptr}, // CPU batched scalar callback function defining the time-dependent coefficient associated with this operator product
                          {hCoefBatchGradComplex64, CUDENSITYMAT_CALLBACK_DEVICE_CPU, nullptr})); // CPU batched scalar gradient callback function defining the gradient of the coefficient with respect to the parameter Omega
    }
    //  Create an empty operator term
    HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle,
                        spaceShape.size(),                   // Hilbert space rank (number of modes)
                        spaceShape.data(),                   // Hilbert space shape (mode extents)
                        &twoBodyTerm));                      // the created empty operator term
    //  Define the operator term: H2 = f(t) * sum_{i < j} {g_ij * ZZ_ij}
    for (int32_t i = 0; i < spaceShape.size() - 1; ++i) {
      for (int32_t j = (i + 1); j < spaceShape.size(); ++j) {
        const double g_ij = -1.0 / static_cast<double>(i + j + 1); // assign some value to the time-independent g_ij coefficient
        HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(handle,
                            twoBodyTerm,
                            1,                                                              // number of elementary tensor operators in the product
                            std::vector<cudensitymatElementaryOperator_t>({spinZZ}).data(), // elementary tensor operators forming the product
                            std::vector<int32_t>({i, j}).data(),                            // space modes acted on by the operator product
                            std::vector<int32_t>({0, 0}).data(),                            // space mode action duality (0: from the left; 1: from the right)
                            make_cuDoubleComplex(g_ij, 0.0),                                // g_ij static coefficient: Always 64-bit-precision complex number
                            cudensitymatScalarCallbackNone,                                 // no time-dependent coefficient associated with this operator product
                            cudensitymatScalarGradientCallbackNone));                       // no coefficient gradient associated with this operator product
      }
    }
    //  Create an empty operator term
    HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle,
                        spaceShape.size(),                   // Hilbert space rank (number of modes)
                        spaceShape.data(),                   // Hilbert space shape (mode extents)
                        &noiseTerm));                        // the created empty operator term
    //  Define the operator term: D1 = d * sum_{i} {YY_ii}
    for (int32_t i = 0; i < spaceShape.size(); ++i) {
      HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(handle,
                          noiseTerm,
                          1,                                                              // number of elementary tensor operators in the product
                          std::vector<cudensitymatElementaryOperator_t>({spinYY}).data(), // elementary tensor operators forming the product
                          std::vector<int32_t>({i, i}).data(),                            // space modes acted on by the operator product (from different sides)
                          std::vector<int32_t>({0, 1}).data(),                            // space mode action duality (0: from the left; 1: from the right)
                          make_cuDoubleComplex(1.0, 0.0),                                 // default coefficient: Always 64-bit-precision complex number
                          cudensitymatScalarCallbackNone,                                 // no time-dependent coefficient associated with this operator product
                          cudensitymatScalarGradientCallbackNone));                       // no coefficient gradient associated with this operator product
    }

    // Construct the full Liouvillian operator as a sum of the created operator terms
    //  Create an empty operator (super-operator)
    HANDLE_CUDM_ERROR(cudensitymatCreateOperator(handle,
                        spaceShape.size(),                // Hilbert space rank (number of modes)
                        spaceShape.data(),                // Hilbert space shape (modes extents)
                        &liouvillian));                   // the created empty operator (super-operator)
    //  Append an operator term to the operator (super-operator)
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                        liouvillian,
                        oneBodyTerm,                      // appended operator term
                        0,                                // operator term action duality as a whole (0: acting from the left; 1: acting from the right)
                        make_cuDoubleComplex(0.0, -1.0),  // -i constant
                        cudensitymatScalarCallbackNone,   // no time-dependent coefficient associated with the operator term as a whole
                        cudensitymatScalarGradientCallbackNone)); // no coefficient gradient associated with the operator term as a whole
    //  Append an operator term to the operator (super-operator)
    fCoefsStaticMinus = static_cast<cuDoubleComplex *>(createInitializeArrayGPU(
      std::vector<cuDoubleComplex>(operBatchSize, make_cuDoubleComplex(0.0, -1.0)))); // -i constant for all coefficient instances in the batch
    fCoefsTotalMinus = static_cast<cuDoubleComplex *>(createInitializeArrayGPU(
      std::vector<cuDoubleComplex>(operBatchSize, make_cuDoubleComplex(0.0, 0.0)))); // storage for the total coefficients for all instances of the batch
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTermBatch(handle,
                        liouvillian,
                        twoBodyTerm,                      // appended operator term
                        0,                                // operator term action duality as a whole (0: acting from the left; 1: acting from the right)
                        operBatchSize,                    // number of instances of the operator term in the batch (they differ by the coefficient value)
                        fCoefsStaticMinus,                // static part of the f(t) batched coefficients in the two-body term
                        fCoefsTotalMinus,                 // total f(t) batched coefficients in the two-body term
                        {fCoefBatchComplex64, CUDENSITYMAT_CALLBACK_DEVICE_CPU, nullptr}, // CPU batched scalar callback function defining the time-dependent coefficient associated with this operator term as a whole
                        {fCoefBatchGradComplex64, CUDENSITYMAT_CALLBACK_DEVICE_CPU, nullptr})); // CPU batched scalar gradient callback function defining the gradient of the coefficient with respect to parameter Omega
    //  Append an operator term to the operator (super-operator)
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                        liouvillian,
                        oneBodyTerm,                      // appended operator term
                        1,                                // operator term action duality as a whole (0: acting from the left; 1: acting from the right)
                        make_cuDoubleComplex(0.0, +1.0),  // +i constant
                        cudensitymatScalarCallbackNone,   // no time-dependent coefficient associated with the operator term as a whole
                        cudensitymatScalarGradientCallbackNone)); // no coefficient gradient associated with the operator term as a whole
    //  Append an operator term to the operator (super-operator)
    fCoefsStaticPlus = static_cast<cuDoubleComplex *>(createInitializeArrayGPU(
      std::vector<cuDoubleComplex>(operBatchSize, make_cuDoubleComplex(0.0, +1.0)))); // +i constant for all coefficient instances in the batch
    fCoefsTotalPlus = static_cast<cuDoubleComplex *>(createInitializeArrayGPU(
      std::vector<cuDoubleComplex>(operBatchSize, make_cuDoubleComplex(0.0, 0.0)))); // storage for the total coefficients for all instances of the batch
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTermBatch(handle,
                        liouvillian,
                        twoBodyTerm,                      // appended operator term
                        1,                                // operator term action duality as a whole (0: acting from the left; 1: acting from the right)
                        operBatchSize,                    // number of instances of the operator term in the batch (they differ by the coefficient value)
                        fCoefsStaticPlus,                 // static part of the f(t) batched coefficients in the dual two-body term
                        fCoefsTotalPlus,                  // total f(t) batched coefficients in the dual two-body term
                        {fCoefBatchComplex64, CUDENSITYMAT_CALLBACK_DEVICE_CPU, nullptr}, // CPU batched scalar callback function defining the time-dependent coefficient associated with this operator term as a whole
                        {fCoefBatchGradComplex64, CUDENSITYMAT_CALLBACK_DEVICE_CPU, nullptr})); // CPU batched scalar gradient callback function defining the gradient of the coefficient with respect to parameter Omega
    //  Append an operator term to the operator (super-operator)
    const double d = 1.0; // assign some value to the time-independent coefficient
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                        liouvillian,
                        noiseTerm,                        // appended operator term
                        0,                                // operator term action duality as a whole (no duality reversing in this case)
                        make_cuDoubleComplex(d, 0.0),     // static coefficient associated with the operator term as a whole
                        cudensitymatScalarCallbackNone,   // no time-dependent coefficient associated with the operator term as a whole
                        cudensitymatScalarGradientCallbackNone)); // no coefficient gradient associated with the operator term as a whole
  }

  // Destructor destructs the user-defined Liouvillian operator
  ~UserDefinedLiouvillian()
  {
    // Destroy the Liouvillian operator
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(liouvillian));

    // Destroy operator terms
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(noiseTerm));
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(twoBodyTerm));
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(oneBodyTerm));

    // Destroy elementary tensor operators
    HANDLE_CUDM_ERROR(cudensitymatDestroyElementaryOperator(spinYY));
    HANDLE_CUDM_ERROR(cudensitymatDestroyElementaryOperator(spinZZ));
    HANDLE_CUDM_ERROR(cudensitymatDestroyElementaryOperator(spinX));

    // Destroy the batched coefficients
    destroyArrayGPU(fCoefsTotalPlus);
    destroyArrayGPU(fCoefsStaticPlus);
    destroyArrayGPU(fCoefsTotalMinus);
    destroyArrayGPU(fCoefsStaticMinus);
    destroyArrayGPU(hCoefsTotal);
    destroyArrayGPU(hCoefsStatic);

    // Destroy operator tensors
    destroyArrayGPU(spinYYelems);
    destroyArrayGPU(spinZZelems);
    destroyArrayGPU(spinXelems);
  }

  // Disable copy constructor/assignment (GPU resources are private, no deep copy)
  UserDefinedLiouvillian(const UserDefinedLiouvillian &) = delete;
  UserDefinedLiouvillian & operator=(const UserDefinedLiouvillian &) = delete;
  UserDefinedLiouvillian(UserDefinedLiouvillian &&) = delete;
  UserDefinedLiouvillian & operator=(UserDefinedLiouvillian &&) = delete;

  /** Returns the number of externally provided Hamiltonian parameters. */
  int32_t getNumParameters() const
  {
    return 1; // one parameter Omega
  }

  /** Get access to the constructed Liouvillian operator. */
  cudensitymatOperator_t & get()
  {
    return liouvillian;
  }

};
