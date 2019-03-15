#pragma once

#include <cstdlib>
#include <unordered_map>

#include "dynet/expr.h"
#include "dynet/dynet.h"

#include "dependency.h"
#include "nn/rnn.h"
#include "nn/arc.h"
#include "activation_function.h"

struct NeuralHeadTaggerSettings
{
    unsigned n_word;
    unsigned n_pos;

    unsigned word_dim = 100;

    unsigned pos_dim = 0;
    bool pos_input = false;
    bool predict_spine = false;
    unsigned n_spine = 0;

    int word_unknown;
    ActivationFunction activation_function;

    template<class Archive> void serialize(Archive& ar, const unsigned int version)
    {
        ar & n_word;
        ar & n_pos;

        ar & word_dim;

        ar & word_unknown;

        ar & activation_function;

        if(version > 0)
        {
            ar & pos_dim;
            ar & pos_input;
            ar & predict_spine;
            ar & n_spine;
        }
        else
        {
            pos_input = false;
            predict_spine = false;
        }
    }
};
BOOST_CLASS_VERSION(NeuralHeadTaggerSettings, 1)

template<class RNNBuilder>
struct NeuralHeadTagger
{
    public:
        const NeuralHeadTaggerSettings settings;

        RNN<RNNBuilder> pos_rnn;

        dynet::Parameter p_hidden_layer_word;
        dynet::Parameter  p_hidden_bias;

        dynet::LookupParameter lp_word;
        dynet::LookupParameter lp_pos;

    public:
        std::unordered_map<unsigned, unsigned> m_word_count;

    explicit NeuralHeadTagger(
        dynet::Model &model, 
        NeuralHeadTaggerSettings t_settings,
        RNNSettings pos_rnn_settings
    ) :
        settings(t_settings),
        pos_rnn(model, pos_rnn_settings, (settings.pos_input ? settings.word_dim + settings.pos_dim : settings.word_dim))
    {
        lp_word = model.add_lookup_parameters(settings.n_word, {settings.word_dim});

        unsigned output_dim = (settings.predict_spine ? settings.n_spine : settings.n_pos);

        p_hidden_layer_word = model.add_parameters({output_dim + 1, pos_rnn.output_dim()});
        p_hidden_bias = model.add_parameters({output_dim + 1});

        if (settings.pos_input)
            lp_pos = model.add_lookup_parameters(settings.n_pos, {settings.pos_dim});
    }

    template<typename Op>
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

        // word embeddings for pos
        std::vector<dynet::expr::Expression> pos_word_embeddings;
        pos_rnn.build(cg, word_embeddings, pos_word_embeddings, dropout, dropout_p);

        dynet::expr::Expression e_hidden_layer_word = parameter(cg, p_hidden_layer_word);
        dynet::expr::Expression e_hidden_bias = parameter(cg, p_hidden_bias);

        // cache softmax output
        std::vector<dynet::expr::Expression> cache_word;
        for (unsigned int i = 0 ; i < word_embeddings.size(); ++i)
        {
            dynet::expr::Expression expr = 
                e_hidden_layer_word * pos_word_embeddings.at(i)
                + 
                e_hidden_bias
            ;
            op(
                i,
                expr
            );
        }
    }
};



