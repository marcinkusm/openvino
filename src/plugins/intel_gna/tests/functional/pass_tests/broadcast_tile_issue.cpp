// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/opsets/opset9.hpp>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace {
struct FunctionConfig {
    std::vector<size_t> input_shape;
    ::ngraph::element::Type ngraph_precision;
};

struct FunctionWithExpect {
    std::shared_ptr<ngraph::Function> function;
    std::vector<std::string> input_friendly_names;
    std::vector<std::string> ouput_friendly_names;
};

class FunctionExpectCreator {
public:
    FunctionExpectCreator(std::string function_name) : _function_name(std::move(function_name)) {}

    virtual ~FunctionExpectCreator() = default;

    std::string GetFunctionName() const;
    virtual FunctionWithExpect CreateFunctionWithExpects(const FunctionConfig& config) const = 0;

private:
    std::string _function_name;
};

inline std::string FunctionExpectCreator::GetFunctionName() const {
    return _function_name;
}

class BroadcastAfterActivationCreator : public FunctionExpectCreator {
public:
    BroadcastAfterActivationCreator() : FunctionExpectCreator("BroadcastAfterActivationCreator") {}

    FunctionWithExpect CreateFunctionWithExpects(const FunctionConfig& config) const override {
        const std::string input_friendly_name = "input_1";
        const auto& input_shape = config.input_shape;
        auto input_param =
            std::make_shared<ngraph::opset9::Parameter>(config.ngraph_precision, ngraph::Shape{input_shape});
        input_param->set_friendly_name(input_friendly_name);

        auto target_shape = ngraph::opset9::Constant::create(::ngraph::element::Type_t::i32,
                                                             ngraph::Shape{{input_shape.size()}},
                                                             input_shape);

        auto length = std::accumulate(input_shape.begin(), input_shape.end(), (size_t)1, std::multiplies<size_t>());
        std::vector<float> vector_data(length, 1.0);
        auto constant = std::make_shared<ngraph::opset9::Constant>(config.ngraph_precision,
                                                                   ngraph::Shape{input_shape},
                                                                   vector_data);
        auto add = std::make_shared<ngraph::opset9::Add>(input_param, constant);
        auto activation = std::make_shared<ngraph::opset9::Sigmoid>(add);

        auto broadcast = std::make_shared<ngraph::opset9::Broadcast>(activation, target_shape);
        const std::string output_friendly_name = "ouput_1";
        broadcast->set_friendly_name(output_friendly_name);
        auto result = std::make_shared<ngraph::opset9::Result>(broadcast);

        auto function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                           ngraph::ParameterVector{input_param},
                                                           GetFunctionName());
        return {function, {input_friendly_name}, {output_friendly_name}};
    }
};

class BroadcastBeforeActivationCreator : public FunctionExpectCreator {
public:
    BroadcastBeforeActivationCreator() : FunctionExpectCreator("BroadcastBeforeActivationCreator") {}

    FunctionWithExpect CreateFunctionWithExpects(const FunctionConfig& config) const override {
        const std::string input_friendly_name = "input_1";
        const auto& input_shape = config.input_shape;
        auto input_param =
            std::make_shared<ngraph::opset9::Parameter>(config.ngraph_precision, ngraph::Shape{input_shape});
        input_param->set_friendly_name(input_friendly_name);

        auto target_shape = ngraph::opset9::Constant::create(::ngraph::element::Type_t::i32,
                                                             ngraph::Shape{{input_shape.size()}},
                                                             input_shape);

        auto broadcast = std::make_shared<ngraph::opset9::Broadcast>(input_param, target_shape);

        auto activation = std::make_shared<ngraph::opset9::Sigmoid>(broadcast);
        const std::string output_friendly_name = "ouput_1";

        auto result = std::make_shared<ngraph::opset9::Result>(activation);

        auto function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                           ngraph::ParameterVector{input_param},
                                                           GetFunctionName());
        return {function, {input_friendly_name}, {output_friendly_name}};
    }
};

class BroadcastTwoOutputsFunctionCreator : public FunctionExpectCreator {
public:
    BroadcastTwoOutputsFunctionCreator() : FunctionExpectCreator("BroadcastTwoOutputsFunctionCreator") {}

    FunctionWithExpect CreateFunctionWithExpects(const FunctionConfig& config) const override {
        const std::string input_friendly_name = "input_1";
        const auto& input_shape = config.input_shape;
        auto input_param =
            std::make_shared<ngraph::opset9::Parameter>(config.ngraph_precision, ngraph::Shape{input_shape});
        input_param->set_friendly_name(input_friendly_name);

        auto target_shape = ngraph::opset9::Constant::create(::ngraph::element::Type_t::i32,
                                                             ngraph::Shape{{input_shape.size()}},
                                                             input_shape);

        auto broadcast = std::make_shared<ngraph::opset9::Broadcast>(input_param, target_shape);
        const std::string output_friendly_name_1 = "ouput_1";
        broadcast->set_friendly_name(output_friendly_name_1);
        auto result_1 = std::make_shared<ngraph::opset9::Result>(broadcast);
        auto length = std::accumulate(input_shape.begin(), input_shape.end(), (size_t)1, std::multiplies<size_t>());
        std::vector<float> vector_data(length, 1.0);
        auto constant = std::make_shared<ngraph::opset9::Constant>(config.ngraph_precision,
                                                                   ngraph::Shape{input_shape},
                                                                   vector_data);
        auto add = std::make_shared<ngraph::opset9::Add>(input_param, constant);
        auto activation = std::make_shared<ngraph::opset9::Sigmoid>(add);
        const std::string output_friendly_name_2 = "ouput_2";
        activation->set_friendly_name(output_friendly_name_2);
        auto result_2 = std::make_shared<ngraph::opset9::Result>(activation);

        ngraph::ResultVector results = {result_1, result_2};
        auto function =
            std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{input_param}, GetFunctionName());
        return {function, {input_friendly_name}, {output_friendly_name_1, output_friendly_name_2}};
    }
};

using TestConfig = std::tuple<std::shared_ptr<FunctionExpectCreator>,  // variant of fuction
                              std::vector<size_t>,                     // input_shape
                              InferenceEngine::Precision               // net precision
                              >;

class BroadcastToTileIssue : public ::testing::WithParamInterface<TestConfig>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TestConfig>& obj);

protected:
    void SetUp() override;
    void Validate() override;
    static const char* s_target_device_name;
    FunctionWithExpect _function_with_expects;
};

const char* BroadcastToTileIssue::s_target_device_name = CommonTestUtils::DEVICE_GNA;

std::string BroadcastToTileIssue::getTestCaseName(const testing::TestParamInfo<TestConfig>& obj) {
    std::shared_ptr<FunctionExpectCreator> function_creator;
    std::vector<size_t> input_shape;
    InferenceEngine::Precision net_precision;

    std::tie(function_creator, input_shape, net_precision) = obj.param;
    std::stringstream test_name;

    test_name << "FunctinVariant=" << function_creator->GetFunctionName() << "_";
    test_name << "IS=" << CommonTestUtils::vec2str(input_shape) << "_";
    test_name << "netPRC=" << net_precision.name() << "_";
    test_name << "targetDevice=" << s_target_device_name << "_";
    return test_name.str();
}

void BroadcastToTileIssue::SetUp() {
    targetDevice = s_target_device_name;

    std::shared_ptr<FunctionExpectCreator> function_creator;
    std::vector<size_t> input_shape;
    InferenceEngine::Precision net_precision;

    std::tie(function_creator, input_shape, net_precision) = GetParam();

    auto ngraph_precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(net_precision);

    FunctionConfig func_config = {input_shape, ngraph_precision};

    _function_with_expects = function_creator->CreateFunctionWithExpects(func_config);
    function = _function_with_expects.function;
}

void BroadcastToTileIssue::Validate() {
    LayerTestsCommon::Validate();
    auto inputs = executableNetwork.GetInputsInfo();
    ASSERT_EQ(_function_with_expects.input_friendly_names.size(), inputs.size());
    for (const auto& name : _function_with_expects.input_friendly_names) {
        ASSERT_TRUE(inputs.end() != inputs.find(name));
    }

    auto outputs = executableNetwork.GetOutputsInfo();
    ASSERT_EQ(_function_with_expects.ouput_friendly_names.size(), outputs.size());
    for (const auto& name : _function_with_expects.ouput_friendly_names) {
        ASSERT_TRUE(outputs.end() != outputs.find(name));
    }
}

TEST_P(BroadcastToTileIssue, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_broadcast_tile_issue,
                         BroadcastToTileIssue,
                         ::testing::Combine(::testing::Values(std::make_shared<BroadcastAfterActivationCreator>(),
                                                              std::make_shared<BroadcastBeforeActivationCreator>(),
                                                              std::make_shared<BroadcastTwoOutputsFunctionCreator>()),
                                            ::testing::Values(std::vector<size_t>{1, 590}),
                                            ::testing::Values(InferenceEngine::Precision::FP32)),
                         BroadcastToTileIssue::getTestCaseName);
}  // namespace
