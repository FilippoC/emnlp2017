#pragma once

struct NNArcSettings
{
    unsigned hidden_units = 100;

    template<class Archive> void serialize(Archive& ar, const unsigned int)
    {
        ar & hidden_units;
    }
};

struct NNArc
{
    const NNArcSettings settings;
    //dynet::LookupParameter* lp_pos;

    dynet::Parameter p_hidden_layer_head_word;
    //dynet::Parameter p_hidden_layer_head_pos;
    dynet::Parameter p_hidden_layer_mod_word;
    //dynet::Parameter p_hidden_layer_mod_pos;
    dynet::Parameter  p_hidden_bias;
    dynet::Parameter p_output_layer;

    dynet::Parameter p_pos_correlation;

    unsigned n_pos;

    NNArc(dynet::Model &model, const NNArcSettings& t_settings, unsigned word_dim, unsigned t_n_pos)
        : settings(t_settings), n_pos(t_n_pos)
    {
        p_hidden_layer_head_word = model.add_parameters({settings.hidden_units, word_dim});
        //p_hidden_layer_head_pos = model.add_parameters({settings.hidden_units, pos_dim});
        p_hidden_layer_mod_word = model.add_parameters({settings.hidden_units, word_dim});
        //p_hidden_layer_mod_pos = model.add_parameters({settings.hidden_units, pos_dim});
        p_hidden_bias = model.add_parameters({settings.hidden_units});
        p_output_layer = model.add_parameters({1, settings.hidden_units});

        p_pos_correlation = model.add_parameters({n_pos, n_pos});
    }

    template<class ArcContainer, class Op>
    void compute(dynet::ComputationGraph& cg, const ArcContainer& arcs, Op op, std::vector<dynet::expr::Expression>& word_embeddings)
    {
        dynet::expr::Expression e_hidden_layer_head_word = parameter(cg, p_hidden_layer_head_word);
        //dynet::expr::Expression e_hidden_layer_head_pos = parameter(cg, p_hidden_layer_head_pos);
        dynet::expr::Expression e_hidden_layer_mod_word  = parameter(cg, p_hidden_layer_mod_word);
        //dynet::expr::Expression e_hidden_layer_mod_pos = parameter(cg, p_hidden_layer_mod_pos);
        dynet::expr::Expression e_hidden_bias = parameter(cg, p_hidden_bias);

        // TODO: We don't have output bias. Is that a problem ?
        dynet::expr::Expression e_output_layer = parameter(cg, p_output_layer);

        dynet::expr::Expression e_pos_correlation = parameter(cg, p_pos_correlation);

        // Cache common operations
        std::vector<dynet::expr::Expression> cache_head_word;
        std::vector<dynet::expr::Expression> cache_mod_word;
        for (unsigned int i = 0 ; i < word_embeddings.size() ; ++i)
        {
            cache_head_word.push_back(e_hidden_layer_head_word * word_embeddings.at(i));
            cache_mod_word.push_back(e_hidden_layer_mod_word * word_embeddings.at(i));
        }

        /*
        std::vector<dynet::expr::Expression> cache_head_pos;
        std::vector<dynet::expr::Expression> cache_mod_pos;
        for (unsigned i = 0 ; i < n_pos ; ++i)
        {
            cache_head_pos.push_back(e_hidden_layer_head_pos * lookup(cg, *lp_pos, i));
            cache_mod_pos.push_back(e_hidden_layer_mod_pos * lookup(cg, *lp_pos, i));
        }
        */


        for (auto const& arc : arcs)
        {
            auto const head = arc.source;
            //auto const head_pos = arc.source_node;
            auto const modifier = arc.destination;
            //auto const modifier_pos = arc.destination_node;


            dynet::expr::Expression output = 
                e_output_layer
                *
                dynet::expr::tanh(
                    cache_head_word.at(head)
                    //+
                    //cache_head_pos.at(head_pos)
                    + 
                    cache_mod_word.at(modifier)
                    //+
                    //cache_mod_pos.at(modifier_pos)
                    + 
                    e_hidden_bias
                )
                +
                // Will probably not compile...
                pick(pick(e_pos_correlation, arc.source_node), arc.destination_node)
            ;

            auto score = as_scalar(cg.get_value(output.i));
            op(arc, score, output);
        }
    }
};

