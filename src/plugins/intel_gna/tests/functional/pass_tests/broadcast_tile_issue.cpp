// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/opsets/opset9.hpp>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace {

enum class FunctionVariant { BroadcastAfterActivation, BroadcastBeforeActivation };

std::map<FunctionVariant, std::string> gc_function_variant_names_map = {
    {FunctionVariant::BroadcastAfterActivation, "BroadcastAfterActivation"},  // checks if there is no regression
    {FunctionVariant::BroadcastBeforeActivation,
     "BroadcastBeforeActivation"}};  // checks fix ConvertTileToLegacyMatcher

struct FunctionConfig {
    FunctionVariant variant;
    std::vector<size_t> input_shape;
    ::ngraph::element::Type ngraph_precision;
    std::string input_name;
    std::string output_name;
    std::string function_name;
};

std::shared_ptr<ngraph::Function> createFunction(const FunctionConfig& config) {
    const auto& input_shape = config.input_shape;
    auto input_param = std::make_shared<ngraph::opset9::Parameter>(config.ngraph_precision, ngraph::Shape{input_shape});
    input_param->set_friendly_name(config.input_name);

    std::shared_ptr<ngraph::opset9::Result> result;
    auto target_shape = ngraph::opset9::Constant::create(::ngraph::element::Type_t::i32,
                                                         ngraph::Shape{{input_shape.size()}},
                                                         input_shape);

    // Broadcast is converted to tile. Tile is converted to Tile IE.
    if (config.variant == FunctionVariant::BroadcastAfterActivation) {
        auto length = std::accumulate(input_shape.begin(), input_shape.end(), (size_t)1, std::multiplies<size_t>());
        std::vector<float> vector_data(length, 1.0);
        auto constant = std::make_shared<ngraph::opset9::Constant>(config.ngraph_precision,
                                                                   ngraph::Shape{input_shape},
                                                                   vector_data);
        auto add = std::make_shared<ngraph::opset9::Add>(input_param, constant);
        auto activation = std::make_shared<ngraph::opset9::Sigmoid>(add);

        auto broadcast = std::make_shared<ngraph::opset9::Broadcast>(activation, target_shape);
        broadcast->set_friendly_name(config.output_name);
        result = std::make_shared<ngraph::opset9::Result>(broadcast);

    } else {
        auto broadcast = std::make_shared<ngraph::opset9::Broadcast>(input_param, target_shape);

        auto activation = std::make_shared<ngraph::opset9::Sigmoid>(broadcast);
        activation->set_friendly_name(config.output_name);
        result = std::make_shared<ngraph::opset9::Result>(activation);
    }

    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_param},
                                              config.function_name);
}

using TestConfig = std::tuple<FunctionVariant,            // variant of fuction
                              std::vector<size_t>,        // input_shape
                              InferenceEngine::Precision  // net precision
                              >;

class BroadcastToTileIssue : public ::testing::WithParamInterface<TestConfig>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TestConfig>& obj);

protected:
    void SetUp() override;
    void Validate() override;
    static const char* s_input_friendly_name;
    static const char* s_output_friendly_name;
    static const char* s_function_name;
    static const char* s_target_device_name;
};

const char* BroadcastToTileIssue::s_input_friendly_name = "test_input_1";
const char* BroadcastToTileIssue::s_output_friendly_name = "test_output_1";
const char* BroadcastToTileIssue::s_function_name = "broadcast_with_activation";
const char* BroadcastToTileIssue::s_target_device_name = CommonTestUtils::DEVICE_GNA;

std::string BroadcastToTileIssue::getTestCaseName(const testing::TestParamInfo<TestConfig>& obj) {
    FunctionVariant function_variant;
    std::vector<size_t> input_shape;
    InferenceEngine::Precision net_precision;

    std::tie(function_variant, input_shape, net_precision) = obj.param;
    std::stringstream test_name;

    test_name << "FunctinVariant=" << gc_function_variant_names_map[function_variant] << "_";
    test_name << "IS=" << CommonTestUtils::vec2str(input_shape) << "_";
    test_name << "netPRC=" << net_precision.name() << "_";
    test_name << "targetDevice=" << s_target_device_name << "_";
    return test_name.str();
}

void BroadcastToTileIssue::SetUp() {
    targetDevice = s_target_device_name;

    FunctionVariant function_variant;
    std::vector<size_t> input_shape;
    InferenceEngine::Precision net_precision;

    std::tie(function_variant, input_shape, net_precision) = GetParam();

    auto ngraph_precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(net_precision);

    FunctionConfig func_config = {function_variant,
                                  input_shape,
                                  ngraph_precision,
                                  s_input_friendly_name,
                                  s_output_friendly_name,
                                  s_function_name};

    function = createFunction(func_config);
}

void BroadcastToTileIssue::Validate() {
    LayerTestsCommon::Validate();
    auto inputs = executableNetwork.GetInputsInfo();
    ASSERT_EQ(1, inputs.size());
    ASSERT_TRUE(inputs.end() != inputs.find(s_input_friendly_name));
    auto outputs = executableNetwork.GetOutputsInfo();
    ASSERT_EQ(1, outputs.size());
    ASSERT_TRUE(outputs.end() != outputs.find(s_output_friendly_name));
}

TEST_P(BroadcastToTileIssue, CompareWithRefs) {
    Run();
}

const std::vector<FunctionVariant> gc_function_variants = {FunctionVariant::BroadcastAfterActivation,
                                                           FunctionVariant::BroadcastBeforeActivation};

INSTANTIATE_TEST_SUITE_P(smoke_broadcast_tile_issue,
                         BroadcastToTileIssue,
                         ::testing::Combine(::testing::ValuesIn(gc_function_variants),
                                            ::testing::Values(std::vector<size_t>{1, 590}),
                                            ::testing::Values(InferenceEngine::Precision::FP32)),
                         BroadcastToTileIssue::getTestCaseName);
}  // namespace
