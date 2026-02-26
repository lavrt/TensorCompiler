#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "node.hpp"
#include "value.hpp"

namespace tensor_compiler::ir {

struct Graph final {
    std::vector<Node> nodes;
    std::vector<Value> values;

    std::unordered_map<std::string, ValueId> value_by_name;

    std::vector<ValueId> graph_inputs;
    std::vector<ValueId> graph_outputs;

    ValueId GetOrCreateValue(const std::string& name) {
        if (auto it = value_by_name.find(name); it != value_by_name.end()) {
            return it->second;
        }
        ValueId id = values.size();
        values.emplace_back(name);
        value_by_name.emplace(values.back().name, id);
        return id;
    }

    ValueId CreateValue(const std::string& name) {
        if (value_by_name.contains(name)) {
            throw std::runtime_error("Value with name '" + name + "' already exists");
        }
        ValueId id = values.size();
        values.emplace_back(name);
        value_by_name.emplace(values.back().name, id);
        return id;
    }

    std::optional<ValueId> GetValueId(const std::string& name) const {
        if (auto it = value_by_name.find(name); it != value_by_name.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    Value& GetValue(ValueId id) {
        return values.at(id);
    }

    const Value& GetValue(ValueId id) const {
        return values.at(id);
    }

    void MarkGraphInput(ValueId id) {
        if (!values.at(id).is_graph_input) {
            values.at(id).is_graph_input = true;
            graph_inputs.push_back(id);
        }
    }

    void MarkGraphOutput(ValueId id) {
        if (!values.at(id).is_graph_output) {
            values.at(id).is_graph_output = true;
            graph_outputs.push_back(id);
        }
    }

    void MarkInitializer(ValueId id) {
        values.at(id).is_initializer = true;
    }

    NodeId AddNode(Node&& node) {
        NodeId nid = nodes.size();
        nodes.emplace_back(std::move(node));

        for (ValueId in : nodes[nid].inputs) {
            values.at(in).consumers.push_back(nid);
        }

        for (ValueId out : nodes[nid].outputs) {
            auto& v = values.at(out);
            if (v.producer.has_value()) {
                throw std::runtime_error("Value '" + v.name + "' already has producer");
            }
            v.producer = nid;
        }

        return nid;
    }

    NodeId AddNode(const Node& node) {
        Node tmp = node;
        return AddNode(std::move(tmp));
    }
};

} // namespace tensor_compiler::ir
