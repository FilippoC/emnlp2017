#pragma once

#include "utils.h"

struct NNNode2Settings
{
    unsigned hidden_units = 100;

    template<class Archive> void serialize(Archive& ar, const unsigned int)
    {
        ar & hidden_units;
    }
};

struct NNNode2
{
    const NNNode2Settings settings;
    //dynet::LookupParameter* lp_pos;

    dynet::Parameter p_hidden_layer;
    dynet::Parameter  p_hidden_bias;

    unsigned n_pos;

    NNNode2 (dynet::Model &model, const NNNode2Settings& t_settings, unsigned word_dim, unsigned t_n_pos)
        : settings(t_settings), n_pos(t_n_pos)
    {
        p_hidden_layer = model.add_parameters({t_n_pos, word_dim});
        p_hidden_bias = model.add_parameters({t_n_pos});
    }

    template<class Node2Container, class Op>
    void compute(dynet::ComputationGraph& cg, const Node2Container& nodes, Op op, std::vector<dynet::expr::Expression>& word_embeddings)
    {
        dynet::expr::Expression e_hidden_layer = parameter(cg, p_hidden_layer);
        dynet::expr::Expression e_hidden_bias = parameter(cg, p_hidden_bias);


        std::vector<dynet::expr::Expression> cache_word;
        for (unsigned int i = 0 ; i < word_embeddings.size(); ++i)
        {
            cache_word.push_back(
                dynet::expr::tanh(
                    e_hidden_layer * word_embeddings.at(i)
                    + 
                    e_hidden_bias
                )
            );
        }

        for (auto const& node : nodes)
        {
            auto const word_index = node.cluster;
            auto const pos_index = node.node;

            dynet::expr::Expression output = 
                dynet::expr::pick(cache_word.at(word_index), pos_index)
            ;

            auto score = as_scalar(cg.get_value(output.i));
            op(node, score, output);
        }
    }
};



