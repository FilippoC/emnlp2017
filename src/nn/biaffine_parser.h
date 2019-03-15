#pragma once

#include "dynet/expr.h"
#include "dynet/dynet.h"

#include "dependency.h"
#include "activation_function.h"
#include "nn/rnn.h"

struct NeuralBiaffineParserSettings
{
    unsigned hidden_units = 100;
    unsigned word_dim;
    unsigned n_word;
    int word_unknown;
    ActivationFunction activation_function;
    unsigned pos_dim;
    unsigned n_pos;
    bool pos_input = false;

    template<class Archive> void serialize(Archive& ar, const unsigned int version)
    {
        ar & hidden_units;
        ar & word_dim;
        ar & n_word;
        ar & word_unknown;
        ar & activation_function;

        if(version > 0)
        {
            ar & pos_dim;
            ar & pos_input;
            ar & n_pos;
        }
        else
        {
            pos_input = false;
        }
    }
};
BOOST_CLASS_VERSION(NeuralBiaffineParserSettings, 1)

template<typename RNNBuilder>
struct NeuralBiaffineParser
{
    const NeuralBiaffineParserSettings settings;

    RNN<RNNBuilder> rnn;

    dynet::Parameter p_hidden_layer_head_word;
    dynet::Parameter p_hidden_layer_mod_word;
    dynet::Parameter p_hidden_bias_head;
    dynet::Parameter p_hidden_bias_mod;

    dynet::Parameter p_ba_head_mod;
    dynet::Parameter p_ba_head;
    dynet::Parameter p_ba_mod;
    dynet::Parameter p_ba_bias;

    dynet::Parameter p_root_embedding;
    dynet::LookupParameter lp_word;
    dynet::LookupParameter lp_pos;

    // TODO: useless ?
    unsigned n_pos;

    std::unordered_map<unsigned, unsigned> m_word_count;

    NeuralBiaffineParser(dynet::Model &model, const NeuralBiaffineParserSettings& t_settings, RNNSettings rnn_settings)
        : settings(t_settings), rnn(model, rnn_settings, (settings.pos_input ? settings.word_dim + settings.pos_dim : settings.word_dim))
    {
        p_hidden_layer_head_word = model.add_parameters({settings.hidden_units, rnn.output_dim()});
        p_hidden_layer_mod_word = model.add_parameters({settings.hidden_units, rnn.output_dim()});
        p_hidden_bias_head = model.add_parameters({settings.hidden_units});
        p_hidden_bias_mod = model.add_parameters({settings.hidden_units});

        p_ba_head_mod = model.add_parameters({settings.hidden_units, settings.hidden_units});
        p_ba_head = model.add_parameters({1, settings.hidden_units});
        p_ba_mod = model.add_parameters({1, settings.hidden_units});
        p_ba_bias = model.add_parameters({1});

        p_root_embedding = model.add_parameters({rnn.output_dim()});
        lp_word = model.add_lookup_parameters(settings.n_word, {settings.word_dim});

        if (settings.pos_input)
            lp_pos = model.add_lookup_parameters(settings.n_pos, {settings.pos_dim});
    }

    template<class Op>
    void compute(
        dynet::ComputationGraph& cg, 
        const IntSentence& sentence,
        Op op,
        bool word_dropout = false,
        bool dropout = false,
        double dropout_p = 0.5
    )
    {
        // RNN stuff for building embeddings

        // Lookup for word embeddings
        std::vector<dynet::expr::Expression> word_embeddings;
        word_embeddings.reserve(sentence.size());

        for (const auto& token : sentence)
        {
            auto word = token.word;

            // Dropout
            if (word_dropout)
            {
                double c = m_word_count.at(word);
                if (!(((double) rand() / RAND_MAX) < (c / (0.25 + c))))
                    word = settings.word_unknown;
            }

            if (settings.pos_input)
                word_embeddings.push_back(dynet::expr::concatenate({lookup(cg, lp_word, word), lookup(cg, lp_pos, token.pos)}));
            else
                word_embeddings.push_back(lookup(cg, lp_word, word));
        }

        std::vector<dynet::expr::Expression> rnn_word_embeddings;
        rnn.build(cg, word_embeddings, rnn_word_embeddings, dropout, dropout_p);


        dynet::expr::Expression e_root_embedding = parameter(cg, p_root_embedding);

        dynet::expr::Expression e_hidden_layer_head_word = parameter(cg, p_hidden_layer_head_word);
        dynet::expr::Expression e_hidden_layer_mod_word = parameter(cg, p_hidden_layer_mod_word);
        dynet::expr::Expression e_hidden_bias_head = parameter(cg, p_hidden_bias_head);
        dynet::expr::Expression e_hidden_bias_mod = parameter(cg, p_hidden_bias_mod);

        dynet::expr::Expression e_ba_head_mod = parameter(cg, p_ba_head_mod);
        dynet::expr::Expression e_ba_head = parameter(cg, p_ba_head);
        dynet::expr::Expression e_ba_mod = parameter(cg, p_ba_mod);
        dynet::expr::Expression e_ba_bias = parameter(cg, p_ba_bias);


        // Cache common operations
        std::vector<dynet::expr::Expression> cache_head_word;
        cache_head_word.push_back(e_hidden_layer_head_word * e_root_embedding);

        // unused first element
        std::vector<dynet::expr::Expression> cache_mod_word(1);

        for (unsigned int i = 0 ; i < rnn_word_embeddings.size() ; ++i)
        {
            dynet::expr::Expression head = dynet::expr::affine_transform({e_hidden_bias_head, e_hidden_layer_head_word, rnn_word_embeddings.at(i)});
            dynet::expr::Expression mod = dynet::expr::affine_transform({e_hidden_bias_mod, e_hidden_layer_mod_word, rnn_word_embeddings.at(i)});

            switch(settings.activation_function)
            {
                case ActivationFunction::relu:
                    cache_head_word.push_back(dynet::expr::rectify(head));
                    cache_mod_word.push_back(dynet::expr::rectify(mod));
                    break;
                case ActivationFunction::tanh:
                    cache_head_word.push_back(dynet::expr::tanh(head));
                    cache_mod_word.push_back(dynet::expr::tanh(mod));
                    break;
                case ActivationFunction::sigmoid:
                    cache_head_word.push_back(dynet::expr::logistic(head));
                    cache_mod_word.push_back(dynet::expr::logistic(mod));
                    break;
            }
        }

        for (unsigned head_index = 0 ; head_index <= sentence.size() ; ++head_index)
        {
            for (unsigned mod_index = 1 ; mod_index <= sentence.size() ; ++ mod_index)
            {
                if (head_index == mod_index)
                    continue;

                dynet::expr::Expression head = cache_head_word.at(head_index);
                dynet::expr::Expression mod = cache_mod_word.at(mod_index);

                dynet::expr::Expression output =
                    (
                        dynet::expr::transpose(head)
                        *
                        e_ba_head_mod
                        *
                        mod
                    )
                    +
                    (e_ba_head * head)
                    +
                    (e_ba_mod * mod)
                    +
                    e_ba_bias
                ;

                auto score = as_scalar(cg.get_value(output.i));
                op(head_index, mod_index, score, output);
            }
        }
    }
};



