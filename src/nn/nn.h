#pragma once

#include <cstdlib>
#include <unordered_map>

#include "dynet/expr.h"
#include "dynet/dynet.h"

#include "nn/rnn.h"
#include "nn/arc.h"
#include "nn/node.h"
#include "nn/node2.h"

struct NNSettings
{
    unsigned n_word;
    unsigned n_pos;

    unsigned word_dim = 100;
    //unsigned pos_dim = 25;

    int word_unknown;

    template<class Archive> void serialize(Archive& ar, const unsigned int)
    {
        ar & n_word;
        ar & n_pos;

        ar & word_dim;
        //ar & pos_dim;

        ar & word_unknown;
    }
};

template<class RNNBuilder>
struct NN
{
    public:
        const NNSettings settings;

        RNN<RNNBuilder> shared_rnn;
        RNN<RNNBuilder> dep_rnn;
        RNN<RNNBuilder> pos_rnn;

        NNArc nn_arc;
        //NNNode nn_node;
        NNNode2 nn_node;


        dynet::LookupParameter lp_word;
        //dynet::LookupParameter lp_pos;

    public:
        std::unordered_map<unsigned, unsigned> m_word_count;

    explicit NN(
        dynet::Model &model, 
        NNSettings t_settings,
        RNNSettings shared_rnn_settings,
        RNNSettings dep_rnn_settings,
        RNNSettings pos_rnn_settings,
        NNArcSettings arc_settings,
        NNNode2Settings node_settings
    ) :
        settings(t_settings),
        shared_rnn(model, shared_rnn_settings, settings.word_dim),
        dep_rnn(model, dep_rnn_settings, shared_rnn.output_dim()),
        pos_rnn(model, pos_rnn_settings, shared_rnn.output_dim()),
        nn_arc(model, arc_settings, dep_rnn.output_dim(), settings.n_pos + 1),
        nn_node(model, node_settings, pos_rnn.output_dim(), settings.n_pos + 1)
    {
        lp_word = model.add_lookup_parameters(settings.n_word + 1, {settings.word_dim});
        //lp_pos = model.add_lookup_parameters(settings.n_pos + 1, {settings.pos_dim});

        //nn_arc.lp_pos = &lp_pos;
        //nn_node.lp_pos = &lp_pos;
    }

    template<
        typename ArcOp,
        typename NodeOp,
        typename ArcContainer,
        typename NodeContainer
    >
    void compute_exprs(
        dynet::ComputationGraph& cg, 
        const IntSentence& sentence,
        ArcOp arc_op,
        NodeOp node_op,
        const ArcContainer& arcs,
        const NodeContainer& nodes,
        bool dropout = false
    )
    {
        // RNN stuff for building embeddings

        // Lookup for word embeddings
        std::vector<dynet::expr::Expression> word_embeddings;
        word_embeddings.reserve(sentence.size() + 1);

        word_embeddings.push_back(lookup(cg, lp_word, settings.n_word));
        for (const auto& token : sentence)
        {
            auto word = token.word;

            // Dropout
            if (dropout)
            {
                double c = m_word_count.at(word);
                if (!(((double) rand() / RAND_MAX) < (c / (0.25 + c))))
                    word = settings.word_unknown;
            }
            word_embeddings.push_back(lookup(cg, lp_word, word));
        }

        // word embeddings after a shared BiRNN for both dep and pos
        std::vector<dynet::expr::Expression> shared_word_embeddings;
        shared_rnn.build(cg, word_embeddings, shared_word_embeddings);

        // word embeddings for deps
        std::vector<dynet::expr::Expression> dep_word_embeddings;
        dep_rnn.build(cg, shared_word_embeddings, dep_word_embeddings);

        // word embeddings for pos
        std::vector<dynet::expr::Expression> pos_word_embeddings;
        pos_rnn.build(cg, shared_word_embeddings, pos_word_embeddings);


        // Feedforward part for computing scores !
        nn_arc.compute(cg, arcs, arc_op, dep_word_embeddings);
        nn_node.compute(cg, nodes, node_op, pos_word_embeddings);
    }
};
