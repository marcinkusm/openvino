// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "orientation_helper.hpp"

#include <legacy/ie_layers.h>

intel_dnn_orientation_t GNAPluginNS::helpers::retrieveInputOrientation(
    const std::string& inputName,
    const InferenceEngine::InputsDataMap& inputDataMap,
    const GNAPluginNS::backend::DnnComponents& components) {
    auto inputItr = inputDataMap.find(inputName);
    if (inputItr == inputDataMap.end()) {
        THROW_GNA_EXCEPTION << "Not found input data for input name: " << inputName << "!";
    }
    auto inputsLayers = InferenceEngine::getInputTo(inputItr->second->getInputData());

    if (inputsLayers.size() == 0) {
        THROW_GNA_EXCEPTION << "Not found layer for input: " << inputName << "!";
    }

    // cannot determine if kDnnNonInterleavedOrientation in case of number of input layers is bigger than 1
    if (inputsLayers.size() != 1) {
        THROW_GNA_EXCEPTION << "Don't know how to handle input: " << inputName << " used as input for more than one layer!";
    }

    auto inputLayer = inputsLayers.begin();
    const auto component = components.findComponent(inputLayer->second);
    if (!component) {
        return kDnnInterleavedOrientation;
    }

    if (component->operation != kDnnInterleaveOp && component->operation != kDnnDeinterleaveOp) {
        return kDnnInterleavedOrientation;
    }

    auto layout = inputItr->second->getTensorDesc().getLayout();

    if (layout != InferenceEngine::Layout::NC && layout != InferenceEngine::Layout::CN &&
        layout != InferenceEngine::Layout::NCHW && layout != InferenceEngine::Layout::NHWC) {
        return kDnnInterleavedOrientation;
    }

    auto input_dims = inputItr->second->getTensorDesc().getDims();

    auto component_rows_num = component->num_rows_in;
    auto component_columns_num = component->num_columns_in;

    // check non-interleaved orientation for when N is a second dim
    if (layout == InferenceEngine::Layout::CN && component_rows_num == input_dims[1] &&
        component_columns_num == input_dims[0]) {
        return kDnnNonInterleavedOrientation;
    }

    // check non-interleaved orientation for other cases
    if (component_rows_num == input_dims[0] && component_columns_num == input_dims[1]) {
        return kDnnNonInterleavedOrientation;
    }

    return kDnnInterleavedOrientation;
}

intel_dnn_orientation_t GNAPluginNS::helpers::retrieveOutputOrientation(
    const std::string& outputName,
    const InferenceEngine::OutputsDataMap& outputsDataMap,
    const GNAPluginNS::backend::DnnComponents& components) {
    auto outputItr = outputsDataMap.find(outputName);
    if (outputItr == outputsDataMap.end()) {
        THROW_GNA_EXCEPTION << "Not found output data for input name: " << outputName << "!";
    }

    auto outputLayer = getCreatorLayer(outputItr->second).lock();

    if (!outputLayer) {
        THROW_GNA_EXCEPTION << "Not found layer for output: " << outputName << "!";
    }

    const auto component = components.findComponent(outputLayer);
    if (!component) {
        return kDnnInterleavedOrientation;
    }

    if (component->operation != kDnnInterleaveOp && component->operation != kDnnDeinterleaveOp) {
        return kDnnInterleavedOrientation;
    }

    auto layout = outputItr->second->getTensorDesc().getLayout();

    if (layout != InferenceEngine::Layout::NC && layout != InferenceEngine::Layout::CN &&
        layout != InferenceEngine::Layout::NCHW && layout != InferenceEngine::Layout::NHWC) {
        return kDnnInterleavedOrientation;
    }

    auto input_dims = outputItr->second->getTensorDesc().getDims();

    auto component_rows_num = component->num_rows_out;
    auto component_columns_num = component->num_columns_out;

    // check non-interleaved orientation for when N is the second dimension
    if (layout == InferenceEngine::Layout::CN && component_rows_num == input_dims[1] &&
        component_columns_num == input_dims[0]) {
        return kDnnNonInterleavedOrientation;
    }

    // check non-interleaved orientation when N is the first dimension
    if (component_rows_num == input_dims[0] && component_columns_num == input_dims[1]) {
        return kDnnNonInterleavedOrientation;
    }

    return kDnnInterleavedOrientation;
}
