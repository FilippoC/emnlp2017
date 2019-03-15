#pragma once

#include "dynet/expr.h"
#include "dynet/dynet.h"

#include "dependency.h"
#include "activation_function.h"
#include "nn/rnn.h"

struct NeuralParserSettings
{
    unsigned hidden_units = 100;
    unsigned word_dim;
    unsigned n_word;
    int word_unknown;
    ActivationFunction activation_function;

    template<class Archive> void serialize(Archive& ar, const unsigned int)
    {
        ar & hidden_units;
        ar & word_dim;
        ar & n_word;
        ar & word_unknown;
        ar & activation_function;
    }
};

template<typename RNNBuilder>
struct NeuralParser
{
    const NeuralParserSettings settings;

    RNN<RNNBuilder> rnn;

    dynet::Parameter p_hidden_layer_head_word;
    dynet::Parameter p_hidden_layer_mod_word;
    dynet::Parameter p_hidden_bias;
    dynet::Parameter p_output_layer;
    dynet::Parameter p_root_embedding;

    dynet::LookupParameter lp_word;

    unsigned n_pos;

    std::unordered_map<unsigned, unsigned> m_word_count;

    NeuralParser(dynet::Model &model, const NeuralParserSettings& t_settings, RNNSettings rnn_settings)
        : settings(t_settings), rnn(model, rnn_settings, settings.word_dim)
    {
        lp_word = model.add_lookup_parameters(settings.n_word, {settings.word_dim});

        p_hidden_layer_head_word = model.add_parameters({settings.hidden_units, rnn.output_dim()});
        p_hidden_layer_mod_word = model.add_parameters({settings.hidden_units, rnn.output_dim()});
        p_hidden_bias = model.add_parameters({settings.hidden_units});
        p_output_layer = model.add_parameters({1, settings.hidden_units});

        p_root_embedding = model.add_parameters({rnn.output_dim()});
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
            word_embeddings.push_back(lookup(cg, lp_word, word));
        }

        std::vector<dynet::expr::Expression> rnn_word_embeddings;
        rnn.build(cg, word_embeddings, rnn_word_embeddings, dropout, dropout_p);


        dynet::expr::Expression e_root_embedding = parameter(cg, p_root_embedding);
        dynet::expr::Expression e_hidden_layer_head_word = parameter(cg, p_hidden_layer_head_word);
        dynet::expr::Expression e_hidden_layer_mod_word  = parameter(cg, p_hidden_layer_mod_word);
        dynet::expr::Expression e_hidden_bias = parameter(cg, p_hidden_bias);

        // TODO: We don't have output bias. Is that a problem for probabilistic models ?
        dynet::expr::Expression e_output_layer = parameter(cg, p_output_layer);


        // Cache common operations
        std::vector<dynet::expr::Expression> cache_head_word;
        // unused first element
        std::vector<dynet::expr::Expression> cache_mod_word(1);

        cache_head_word.push_back(e_hidden_layer_head_word * e_root_embedding);
        for (unsigned int i = 0 ; i < rnn_word_embeddings.size() ; ++i)
        {
            cache_head_word.push_back(e_hidden_layer_head_word * rnn_word_embeddings.at(i));
            cache_mod_word.push_back(e_hidden_layer_mod_word * rnn_word_embeddings.at(i));
        }

        for (unsigned head_index = 0 ; head_index <= sentence.size() ; ++head_index)
        {
            for (unsigned mod_index = 1 ; mod_index <= sentence.size() ; ++ mod_index)
            {
                if (head_index == mod_index)
                    continue;

                dynet::expr::Expression e_linear =
                    cache_head_word.at(head_index)
                    + 
                    cache_mod_word.at(mod_index)
                    + 
                    e_hidden_bias
                ;

                dynet::expr::Expression e_nonlinear;
                switch(settings.activation_function)
                {
                    case ActivationFunction::relu:
                        e_nonlinear = dynet::expr::rectify(e_linear);
                        break;
                    case ActivationFunction::tanh:
                        e_nonlinear = dynet::expr::tanh(e_linear);
                        break;
                    case ActivationFunction::sigmoid:
                        e_nonlinear = dynet::expr::logistic(e_linear);
                        break;
                }

                dynet::expr::Expression output = e_output_layer * e_nonlinear;

                auto score = as_scalar(cg.get_value(output.i));
                op(head_index, mod_index, score, output);
            }
        }
    }
};


