// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset2.hpp>
#include <openvino/opsets/opset7.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/gelu7_downgrade.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, Gelu7Downgrade) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 2, 3});
        auto gelu = std::make_shared<opset7::Gelu>(input, op::GeluApproximationMode::ERF);

        model = std::make_shared<ov::Model>(NodeVector{gelu}, ParameterVector{input});

        manager.register_pass<ov::pass::Gelu7Downgrade>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 2, 3});
        auto gelu = std::make_shared<opset2::Gelu>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{gelu}, ParameterVector{input});
    }
}
