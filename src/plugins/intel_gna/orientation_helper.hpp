// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie/ie_input_info.hpp>
#include <string>

#include "backend/dnn_components.hpp"

namespace GNAPluginNS {
/**
 * @namespace helpers contains helpers tools for gna plugin.
 */
namespace helpers {

/**
 * @brief Retrieve expected orientation for input of given \p inputName. It is needed to recognize if extra
 * transposition for input data of input layer is needed.
 *
 * @note Function check following parameters if:
 *  - there is only one input layer
 *  - input tensor layout is one of following layouts NC, CN, NCHW, NHWC
 *  - number of input rows of component of coresponding layer component layer is equal to input tensor N dimension
 *  - number of input column of component of coresponding input layer component is equal to input C dimension
 * If any of condtion above will be not met kDnnInterleavedOrientation is returned by default.
 * If all conditions are met kDnnNonInterleavedOrientation is retuned.
 *
 * @return kDnnNonInterleavedOrientation if condition is met, otherwise return kDnnInterleavedOrientation
 * @throws in case there is no \p inputName in \p inputDataMap or if there is no input layer for input with given \p
 * inputName
 */
intel_dnn_orientation_t retrieveInputOrientation(const std::string& inputName,
                                                 const InferenceEngine::InputsDataMap& inputDataMap,
                                                 const GNAPluginNS::backend::DnnComponents& components);

/**
 * @brief Retrieve expected orientation for output of given \p outputName. It is needed to recognize if extra
 * transposition for output data of output layer is needed.
 *
 * @note Function checks following parameters if:
 *  - output tensor layout is one of following layouts NC, CN, NCHW, NHWC
 *  - number of input rows of component of coresponding output layer component is equal to output tensor N sdimension
 *  - number of input column of component of coresponding output layer component is equal to input C dimension
 * If any of condtion above will be not met kDnnInterleavedOrientation is returned by default.
 * If all conditions are met kDnnNonInterleavedOrientation is retuned.
 *
 * @return kDnnNonInterleavedOrientation if condition is met, otherwise return kDnnInterleavedOrientation
 * @throws in case there is no \p inputName in \p inputDataMap or if there is no input layer for input with given \p
 * inputName
 */
intel_dnn_orientation_t retrieveOutputOrientation(const std::string& outputName,
                                                  const InferenceEngine::OutputsDataMap& outputsDataMap,
                                                  const GNAPluginNS::backend::DnnComponents& components);

}  // namespace helpers
}  // namespace GNAPluginNS
