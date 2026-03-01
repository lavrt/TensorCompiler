#include <string>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include <onnx/onnx_pb.h>

#include "onnx_loader.hpp"
#include "onnx_importer.hpp"

namespace tc = tensor_compiler;

// BranchyModel ------------------------------------------------------------------------------------

TEST(Import, BranchyModel) {
    const std::string path = std::string(TC_SOURCE_DIR) + "/tests/testdata/branchy.onnx";

    onnx::ModelProto model = tc::frontend::LoadOnnxModel(path);
    tc::ir::Graph g = tc::frontend::ImportOnnx(model.graph());

    ASSERT_EQ(g.nodes.size(), 5u);
    auto x = g.GetValueId("x");
    ASSERT_TRUE(x.has_value());
    ASSERT_GE(g.values[*x].consumers.size(), 2u);

    auto b1 = g.GetValueId("b1");
    ASSERT_TRUE(b1.has_value());
    ASSERT_TRUE(g.values[*b1].is_initializer);
}

// AllOpsWithAttrs ---------------------------------------------------------------------------------

namespace {

const tc::ir::Node* FindNodeByName(const tc::ir::Graph& g, const std::string& name) {
    for (const auto& n : g.nodes) {
        if (n.name == name) return &n;
    }
    return nullptr;
}

std::optional<std::int64_t> GetAttrInt(const tc::ir::Node& n, const std::string& key) {
    auto it = n.attrs.find(key);
    if (it == n.attrs.end()) return std::nullopt;
    if (auto p = std::get_if<std::int64_t>(&it->second)) return *p;
    return std::nullopt;
}

std::optional<double> GetAttrDouble(const tc::ir::Node& n, const std::string& key) {
    auto it = n.attrs.find(key);
    if (it == n.attrs.end()) return std::nullopt;
    if (auto p = std::get_if<double>(&it->second)) return *p;
    return std::nullopt;
}

std::optional<std::vector<std::int64_t>> GetAttrInts(const tc::ir::Node& n, const std::string& key) {
    auto it = n.attrs.find(key);
    if (it == n.attrs.end()) return std::nullopt;
    if (auto p = std::get_if<std::vector<std::int64_t>>(&it->second)) return *p;
    return std::nullopt;
}

} // namespace

TEST(Import, AllOpsWithAttrs) {
    const std::string path = std::string(TC_SOURCE_DIR) + "/tests/testdata/all_ops_attrs.onnx";

    onnx::ModelProto model = tc::frontend::LoadOnnxModel(path);
    tc::ir::Graph g = tc::frontend::ImportOnnx(model.graph());

    ASSERT_GE(g.nodes.size(), 8u);

    for (const std::string& w : {"Wc", "bc", "add_const", "mul_const", "Wmm", "Bg", "Cg"}) {
        auto vid = g.GetValueId(w);
        ASSERT_TRUE(vid.has_value()) << "Missing value: " << w;
        EXPECT_TRUE(g.values[*vid].is_initializer) << "Not marked initializer: " << w;

        EXPECT_TRUE(g.constants.contains(*vid)) << "Missing constants entry for: " << w;
    }

    const auto* conv = FindNodeByName(g, "Conv1");
    ASSERT_NE(conv, nullptr);

    auto kshape = GetAttrInts(*conv, "kernel_shape");
    auto strides = GetAttrInts(*conv, "strides");
    auto dil = GetAttrInts(*conv, "dilations");
    auto pads = GetAttrInts(*conv, "pads");
    auto group = GetAttrInt(*conv, "group");

    ASSERT_TRUE(kshape.has_value());
    ASSERT_TRUE(strides.has_value());
    ASSERT_TRUE(dil.has_value());
    ASSERT_TRUE(pads.has_value());
    ASSERT_TRUE(group.has_value());

    EXPECT_EQ(*kshape, (std::vector<std::int64_t>{3, 3}));
    EXPECT_EQ(*strides, (std::vector<std::int64_t>{1, 1}));
    EXPECT_EQ(*dil, (std::vector<std::int64_t>{1, 1}));
    EXPECT_EQ(*pads, (std::vector<std::int64_t>{0, 0, 0, 0}));
    EXPECT_EQ(*group, 1);

    const auto* gemm = FindNodeByName(g, "Gemm1");
    ASSERT_NE(gemm, nullptr);

    auto alpha = GetAttrDouble(*gemm, "alpha");
    auto beta  = GetAttrDouble(*gemm, "beta");
    auto transA = GetAttrInt(*gemm, "transA");
    auto transB = GetAttrInt(*gemm, "transB");

    ASSERT_TRUE(alpha.has_value());
    ASSERT_TRUE(beta.has_value());
    ASSERT_TRUE(transA.has_value());
    ASSERT_TRUE(transB.has_value());

    EXPECT_DOUBLE_EQ(*alpha, 1.0);
    EXPECT_DOUBLE_EQ(*beta, 1.0);
    EXPECT_EQ(*transA, 0);
    EXPECT_EQ(*transB, 1);
}
