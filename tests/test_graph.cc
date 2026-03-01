#include <gtest/gtest.h>

#include "graph.hpp"
#include "node.hpp"

namespace tc = tensor_compiler;

TEST(Graph, LinksProducerConsumers) {
    tc::ir::Graph g;

    auto x = g.GetOrCreateValue("x");
    auto y = g.GetOrCreateValue("y");
    g.MarkGraphInput(x);
    g.MarkGraphOutput(y);

    tc::ir::Node n(
        "Relu1",
        "Relu",
        std::vector<tc::ir::ValueId>{x},
        std::vector<tc::ir::ValueId>{y},
        tc::ir::AttrMap{}
    );
    g.AddNode(std::move(n));

    ASSERT_EQ(g.values[x].consumers.size(), 1u);
    ASSERT_TRUE(g.values[y].producer.has_value());
}

TEST(Graph, GetOrCreateValueReturnsSameIdForSameName) {
    tc::ir::Graph g;

    auto a1 = g.GetOrCreateValue("a");
    auto a2 = g.GetOrCreateValue("a");
    auto b  = g.GetOrCreateValue("b");

    ASSERT_EQ(a1, a2);
    ASSERT_NE(a1, b);
    ASSERT_EQ(g.values.size(), 2u);
}

TEST(Graph, MarkGraphInputOutputDoesNotDuplicateIds) {
    tc::ir::Graph g;

    auto x = g.GetOrCreateValue("x");
    auto y = g.GetOrCreateValue("y");

    g.MarkGraphInput(x);
    g.MarkGraphInput(x);
    g.MarkGraphOutput(y);
    g.MarkGraphOutput(y);

    ASSERT_EQ(g.graph_inputs.size(), 1u);
    ASSERT_EQ(g.graph_outputs.size(), 1u);
    ASSERT_TRUE(g.values[x].is_graph_input);
    ASSERT_TRUE(g.values[y].is_graph_output);
}

TEST(Graph, SingleProducerPerValueIsEnforced) {
    tc::ir::Graph g;

    auto x = g.GetOrCreateValue("x");
    auto y = g.GetOrCreateValue("y");
    auto z = g.GetOrCreateValue("z");

    g.AddNode(tc::ir::Node{
        "N1", "Relu",
        std::vector<tc::ir::ValueId>{x},
        std::vector<tc::ir::ValueId>{y},
        tc::ir::AttrMap{}
    });

    EXPECT_THROW({
        g.AddNode(tc::ir::Node{
            "N2", "Relu",
            std::vector<tc::ir::ValueId>{z},
            std::vector<tc::ir::ValueId>{y},
            tc::ir::AttrMap{}
        });
    }, std::runtime_error);
}

TEST(Graph, ConsumersAndProducerAreLinkedCorrectlyForChain) {
    tc::ir::Graph g;

    auto x = g.GetOrCreateValue("x");
    auto t = g.GetOrCreateValue("t");
    auto y = g.GetOrCreateValue("y");

    g.AddNode(tc::ir::Node{
        "Add1", "Add",
        std::vector<tc::ir::ValueId>{x, x},
        std::vector<tc::ir::ValueId>{t},
        tc::ir::AttrMap{}
    });

    g.AddNode(tc::ir::Node{
        "Relu1", "Relu",
        std::vector<tc::ir::ValueId>{t},
        std::vector<tc::ir::ValueId>{y},
        tc::ir::AttrMap{}
    });

    ASSERT_EQ(g.values[x].consumers.size(), 2u);
    ASSERT_TRUE(g.values[t].producer.has_value());
    ASSERT_EQ(*g.values[t].producer, 0u);
    ASSERT_EQ(g.values[t].consumers.size(), 1u);
    ASSERT_EQ(g.values[t].consumers[0], 1u);
    ASSERT_TRUE(g.values[y].producer.has_value());
    ASSERT_EQ(*g.values[y].producer, 1u);
}

TEST(Graph, BranchingOneValueMultipleConsumers) {
    tc::ir::Graph g;

    auto x  = g.GetOrCreateValue("x");
    auto y1 = g.GetOrCreateValue("y1");
    auto y2 = g.GetOrCreateValue("y2");

    g.AddNode(tc::ir::Node{
        "ReluA", "Relu",
        std::vector<tc::ir::ValueId>{x},
        std::vector<tc::ir::ValueId>{y1},
        tc::ir::AttrMap{}
    });

    g.AddNode(tc::ir::Node{
        "ReluB", "Relu",
        std::vector<tc::ir::ValueId>{x},
        std::vector<tc::ir::ValueId>{y2},
        tc::ir::AttrMap{}
    });

    ASSERT_EQ(g.values[x].consumers.size(), 2u);
    ASSERT_EQ(g.values[x].consumers[0], 0u);
    ASSERT_EQ(g.values[x].consumers[1], 1u);
}

TEST(Graph, MergeTwoProducersIntoOneConsumerViaAdd) {
    tc::ir::Graph g;

    auto x1 = g.GetOrCreateValue("x1");
    auto x2 = g.GetOrCreateValue("x2");
    auto a  = g.GetOrCreateValue("a");
    auto b  = g.GetOrCreateValue("b");
    auto y  = g.GetOrCreateValue("y");

    g.AddNode(tc::ir::Node{
        "Relu1", "Relu",
        std::vector<tc::ir::ValueId>{x1},
        std::vector<tc::ir::ValueId>{a},
        tc::ir::AttrMap{}
    });

    g.AddNode(tc::ir::Node{
        "Relu2", "Relu",
        std::vector<tc::ir::ValueId>{x2},
        std::vector<tc::ir::ValueId>{b},
        tc::ir::AttrMap{}
    });

    g.AddNode(tc::ir::Node{
        "AddMerge", "Add",
        std::vector<tc::ir::ValueId>{a, b},
        std::vector<tc::ir::ValueId>{y},
        tc::ir::AttrMap{}
    });

    ASSERT_EQ(g.values[a].consumers.size(), 1u);
    ASSERT_EQ(g.values[b].consumers.size(), 1u);
    ASSERT_EQ(g.values[a].consumers[0], 2u);
    ASSERT_EQ(g.values[b].consumers[0], 2u);

    ASSERT_TRUE(g.values[y].producer.has_value());
    ASSERT_EQ(*g.values[y].producer, 2u);
}

TEST(Graph, CreateValueThrowsOnDuplicateName) {
    tc::ir::Graph g;

    g.CreateValue("x");
    EXPECT_THROW({ g.CreateValue("x"); }, std::runtime_error);
}
