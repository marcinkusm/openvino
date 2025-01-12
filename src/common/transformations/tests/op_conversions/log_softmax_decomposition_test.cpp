// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset5.hpp>
#include <openvino/pass/manager.hpp>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/log_softmax_decomposition.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, LogSoftmaxDecomposition) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{3, 2});
        auto log_softmax = std::make_shared<opset5::LogSoftmax>(data, 1);

        model = std::make_shared<ov::Model>(NodeVector{log_softmax}, ParameterVector{data});

        manager.register_pass<ov::pass::LogSoftmaxDecomposition>();
    }

    {
        auto input0 = std::make_shared<opset5::Parameter>(element::f32, Shape{3, 2});
        auto axis1_const = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto max = std::make_shared<opset5::ReduceMax>(input0, axis1_const, true);
        auto sub = std::make_shared<opset5::Subtract>(input0, max);
        auto exp = std::make_shared<opset5::Exp>(sub);
        auto axis2_const = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto sum = std::make_shared<opset5::ReduceSum>(exp, axis2_const, true);
        auto log = std::make_shared<opset5::Log>(sum);
        auto sub_end = std::make_shared<opset5::Subtract>(sub, log);

        model_ref = std::make_shared<ov::Model>(NodeVector{sub_end}, ParameterVector{input0});
    }
}
