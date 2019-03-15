#pragma once

struct NNNodeSettings
{
    unsigned hidden_units = 100;

    template<class Archive> void serialize(Archive& ar, const unsigned int)
    {
        ar & hidden_units;
    }
};

struct NNNode
{
    const NNNodeSettings settings;
    dynet::LookupParameter* lp_pos;

    dynet::Parameter p_hidden_layer_word;
    dynet::Parameter p_hidden_layer_pos;
    dynet::Parameter  p_hidden_bias;
    dynet::Parameter p_output_layer;

    unsigned n_pos;

    NNNode (dynet::Model &model, const NNNodeSettings& t_settings, unsigned word_dim, unsigned pos_dim, unsigned t_n_pos)
        : settings(t_settings), n_pos(t_n_pos)
    {
        p_hidden_layer_word = model.add_parameters({settings.hidden_units, word_dim});
        p_hidden_layer_pos = model.add_parameters({settings.hidden_units, pos_dim});
        p_hidden_bias = model.add_parameters({settings.hidden_units});
        p_output_layer = model.add_parameters({1, settings.hidden_units});
    }

    template<class NodeContainer, class Op>
    void compute(dynet::ComputationGraph& cg, const NodeContainer& nodes, Op op, std::vector<dynet::expr::Expression>& word_embeddings)
    {
        dynet::expr::Expression e_hidden_layer_word = parameter(cg, p_hidden_layer_word);
        dynet::expr::Expression e_hidden_layer_pos = parameter(cg, p_hidden_layer_pos);
        dynet::expr::Expression e_hidden_bias = parameter(cg, p_hidden_bias);

        // TODO: We don't have output bias. Is that a problem ?
        dynet::expr::Expression e_output_layer = parameter(cg, p_output_layer);

        // Cache common operations
        std::vector<dynet::expr::Expression> cache_word;
        for (unsigned int i = 0 ; i < word_embeddings.size() ; ++i)
        {
            cache_word.push_back(e_hidden_layer_word * word_embeddings.at(i));
        }

        std::vector<dynet::expr::Expression> cache_pos;
        for (unsigned i = 0 ; i < n_pos ; ++i)
        {
            cache_pos.push_back(e_hidden_layer_pos * lookup(cg, *lp_pos, i));
        }


        for (auto const& node : nodes)
        {
            auto const word_index = node.cluster;
            auto const pos_index = node.node;

            dynet::expr::Expression output = 
                e_output_layer
                *
                dynet::expr::tanh(
                    cache_word.at(word_index)
                    +
                    cache_pos.at(pos_index)
                    + 
                    e_hidden_bias
                )
            ;

            auto score = as_scalar(cg.get_value(output.i));
            op(node, score, output);
        }
    }
};


