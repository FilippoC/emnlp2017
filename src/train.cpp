#include <iostream>
#include <vector>

#include <algorithm>
#include <unordered_map>
#include <set>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <chrono>

#include "dynet/lstm.h"
#include "dynet/training.h"
#include "dynet/dynet.h"
#include "dynet/dict.h"

#include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <boost/program_options.hpp>

#include "decoder.h"
#include "serialization.h"
#include "nn/nn.h"
#include "graph_generator.h"
#include "utils.h"
#include "graph.h"
#include "status.h"

#include "dependency.h"
#include "reader.h"


int main(int argc, char **argv)
{
    std::string train_path;
    std::string dev_path;
    std::string model_path;
    unsigned n_iteration;
    bool use_cpos;
    bool limit_pos_distance;

    bool eval_on_dev;
    unsigned start_eval_at;

    StepsizeOptions stepsize_options;
    unsigned max_iteration;
    bool use_reduction;

    std::string ignore_dynet_mem;
    std::string ignore_dynet_wd;

    unsigned dim_embeddings;
    unsigned lstm_shared_dim;
    unsigned lstm_shared_stack;
    unsigned lstm_shared_layer;
    unsigned lstm_pos_dim;
    unsigned lstm_pos_stack;
    unsigned lstm_pos_layer;
    unsigned lstm_dep_dim;
    unsigned lstm_dep_stack;
    unsigned lstm_dep_layer;
    unsigned nn_pos_hidden_units;
    unsigned nn_dep_hidden_units;

    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
        ("train", po::value<std::string>(&train_path)->required(), "")
        ("model", po::value<std::string>(&model_path)->required(), "")
        ("iteration", po::value<unsigned>(&n_iteration)->default_value(30u), "")
        ("cpos", po::value<bool>(&use_cpos)->default_value(false), "")
        ("limit-pos-distance", po::value<bool>(&limit_pos_distance)->default_value(true), "")
        ("eval-on-dev", po::value<bool>(&eval_on_dev)->default_value(false), "")
        ("start-eval-at", po::value<unsigned>(&start_eval_at)->default_value(0u), "")
        ("dev-path", po::value<std::string>(&dev_path)->default_value(""), "")
        // SGD options for dev
        ("reduction", po::value<bool>(&use_reduction)->default_value(false), "")
        ("max-iteration", po::value<unsigned>(&max_iteration)->default_value(500), "")
        ("stepsize-scale", po::value<double>(&stepsize_options.stepsize_scale)->default_value(1.0), "SGD: stepsize scale")
        ("polyak", po::value<bool>(&stepsize_options.polyak)->default_value(false), "SGD: use polyak steapsize")
        ("polyak-wub", po::value<double>(&stepsize_options.polyak_wub)->default_value(1.0), "SGD: weight of the UB (>= 1.0)")
        ("decreasing", po::value<bool>(&stepsize_options.decreasing)->default_value(true), "SGD: automatiquely decrease stepsize")
        ("constant-decreasing", po::value<bool>(&stepsize_options.constant_decreasing)->default_value(false), "SGD: decrease stepsize at each iteration")
        ("camerini", po::value<bool>(&stepsize_options.camerini)->default_value(false), "SGD: use Camerini et al. momentum subgradient")
        ("gamma", po::value<double>(&stepsize_options.gamma)->default_value(1.5), "SGD: gamma paremeter for Camerini et al. momentum subgradient")
        ("dynet-mem", po::value<std::string>(&ignore_dynet_mem), "")
        ("dynet-weight-decay", po::value<std::string>(&ignore_dynet_wd), "")
        // NN options
        ("dim-embeddings", po::value<unsigned>(&dim_embeddings)->default_value(100))
        ("lstm-shared-dim", po::value<unsigned>(&lstm_shared_dim)->default_value(125))
        ("lstm-shared-stack", po::value<unsigned>(&lstm_shared_stack)->default_value(0))
        ("lstm-shared-layer", po::value<unsigned>(&lstm_shared_layer)->default_value(1))
        ("lstm-pos-dim", po::value<unsigned>(&lstm_pos_dim)->default_value(125))
        ("lstm-pos-stack", po::value<unsigned>(&lstm_pos_stack)->default_value(1))
        ("lstm-pos-layer", po::value<unsigned>(&lstm_pos_layer)->default_value(1))
        ("lstm-dep-dim", po::value<unsigned>(&lstm_dep_dim)->default_value(125))
        ("lstm-dep-stack", po::value<unsigned>(&lstm_dep_stack)->default_value(1))
        ("lstm-dep-layer", po::value<unsigned>(&lstm_dep_layer)->default_value(1))
        ("nn-pos-hidden-units", po::value<unsigned>(&nn_pos_hidden_units)->default_value(100))
        ("nn-dep-hidden-units", po::value<unsigned>(&nn_dep_hidden_units)->default_value(100))
    ;

    po::positional_options_description pod; 
    pod.add("train", 1); 
    pod.add("model", 1); 

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pod).run(), vm);
    po::notify(vm);

    dynet::initialize(argc, argv);


    dynet::Dict word_dict;
    dynet::Dict pos_dict;
    

    std::vector<IntSentence> train_data;
    std::vector<IntSentence> dev_data;

    std::cerr << "Reading train data..." << std::endl << std::flush;
    read_conll_file(
            train_path,
            std::back_inserter(train_data),
            [&] (const StringToken& token) {
                return convert(word_dict, pos_dict, token);
            },
            use_cpos
    );

    word_dict.freeze();
    pos_dict.freeze();

    word_dict.set_unk("*UNKNOWN*");

    if (eval_on_dev)
    {
        std::cerr << "Reading dev data..." << std::endl << std::flush;
        read_conll_file(
                dev_path,
                std::back_inserter(dev_data),
                [&] (const StringToken& token) {
                    return convert(word_dict, pos_dict, token);
                },
                use_cpos
        );
    }

    dynet::Model model;
    dynet::AdamTrainer trainer(model);
    trainer.sparse_updates_enabled = false;

    RNNSettings shared_rnn_settings;
    shared_rnn_settings.dim = lstm_shared_dim;
    shared_rnn_settings.n_stack = lstm_shared_stack;
    shared_rnn_settings.n_layer = lstm_shared_layer;

    RNNSettings arc_rnn_settings;
    arc_rnn_settings.dim = lstm_dep_dim;
    arc_rnn_settings.n_stack = lstm_dep_stack;
    arc_rnn_settings.n_layer = lstm_dep_layer;
    
    RNNSettings node_rnn_settings;
    node_rnn_settings.dim = lstm_pos_dim;
    node_rnn_settings.n_stack = lstm_pos_stack;
    node_rnn_settings.n_layer = lstm_pos_layer;

    NNSettings nn_settings;
    nn_settings.n_word = word_dict.size();
    nn_settings.n_pos = pos_dict.size();
    nn_settings.word_unknown = word_dict.convert("*UNKNOWN*");
    nn_settings.word_dim = dim_embeddings;

    NNArcSettings nn_arc_settings;
    nn_arc_settings.hidden_units = nn_pos_hidden_units;

    NNNode2Settings nn_node_settings;
    nn_node_settings.hidden_units = nn_dep_hidden_units;



    NN<dynet::LSTMBuilder> rnn(model, nn_settings, shared_rnn_settings, arc_rnn_settings, node_rnn_settings, nn_arc_settings, nn_node_settings);
    

    // Allowed POS by word + allowed dependencies between POS
    GraphGenerator<IntSentence> graph_generator(pos_dict.size());
    for (auto const& sentence : train_data)
    {
        graph_generator.update(sentence);
        // Count word frequence (used for dropout)
        for (auto const& token : sentence)
            rnn.m_word_count[token.word] += 1;
    }

    if (eval_on_dev)
        graph_generator.populate_with_everything(word_dict.convert("*UNKNOWN*"));

    double loss = 0.0;

    save_object(model_path + ".word_dict", word_dict);
    save_object(model_path + ".pos_dict", pos_dict);
    save_object(model_path + ".nn_settings", nn_settings);
    save_object(model_path + ".shared_rnn_settings", shared_rnn_settings);
    save_object(model_path + ".arc_rnn_settings", arc_rnn_settings);
    save_object(model_path + ".node_rnn_settings", node_rnn_settings);
    save_object(model_path + ".arc_nn_settings", nn_arc_settings);
    save_object(model_path + ".node_nn_settings", nn_node_settings);
    save_object(model_path + ".graph_generator", graph_generator);

    for (unsigned iteration = 0 ; iteration <= n_iteration ; ++iteration)
    {
        std::cerr << "Iteration: " << iteration << std::endl;
        std::cerr << std::flush;

        unsigned n_correct_head = 0;
        unsigned n_correct_pos = 0;
        unsigned n_total = 0;
        
        std::random_shuffle(std::begin(train_data), std::end(train_data));

        for (auto const& sentence : train_data)
        {
            dynet::ComputationGraph cg;

            Status status;
            status.n_cluster = sentence.size() + 1;

            // TODO: this may create some inaccessible node & arcs
            // => do a reduction step before computing weights ?
            graph_generator.build_arcs(
                    sentence,
                    [&] (const Arc& arc)
                    {
                        status.arcs.push_back(arc);
                    },
                    [&] (const Node& node)
                    {
                        status.nodes.push_back(node);
                    },
                    limit_pos_distance
            );


            std::vector<dynet::expr::Expression> arc_exprs;
            std::vector<dynet::expr::Expression> node_exprs;
            rnn.compute_exprs(
                cg,
                sentence,
                [&] (const Arc& arc, double weight, dynet::expr::Expression& expr) -> void
                {
                    arc_exprs.push_back(expr);

                    status.original_weights.push_back(weight);

                    // loss-augmented inference
                    auto const& token = sentence[arc.destination];
                    if (!arc.is(
                        token.head,
                        token.head == 0 ? nn_settings.n_pos : sentence[token.head].pos,
                        token.index,
                        token.pos
                    ))
                        weight += 1.0;

                    double w = weight / 3.0;
                    status.cmsa_weights.push_back(w);
                    status.incoming_weights.push_back(w);
                    status.outgoing_weights.push_back(w);
                },
                [&] (const Node& node, double weight, dynet::expr::Expression& expr) -> void
                {
                    unused_parameter(expr);

                    node_exprs.push_back(expr);

                    if (node.cluster != 0)
                    {
                        auto const& token = sentence[node.cluster];
                        if (token.pos != node.node)
                            weight += 1.0;
                    }
                    status.node_weights.push_back(weight);
                },
                status.arcs,
                status.nodes,
                true // dropout
            );

            Subgradient subgradient(stepsize_options, status);
            DecoderTimer timer;
            // TODO: use decode_dual instead
            // but then we need to have parameter for loss augmented + word dropout
            bool converged = decode(
                status,
                subgradient,
                max_iteration,
                use_reduction,
                timer
            );

            std::vector<double> arc_outputs(status.arcs.size(), 0.0);
            std::vector<double> node_outputs(status.nodes.size(), 0.0);

            // if converged primal = dual
            if (converged)
            {
                assert(std::isfinite(status.primal_weight));
                std::vector<int> nodes(status.n_cluster);

                for (unsigned i = 0u ; i < status.arcs.size() ; ++i)
                {
                    if (!status.primal_arcs.at(i))
                        continue;

                    arc_outputs[i] = 1.0;
                    auto const arc = status.arcs.at(i);
                    nodes.at(arc.destination) = arc.destination_node;
                }

                for (unsigned i = 0u ; i < status.nodes.size() ; ++i)
                {
                    auto const node = status.nodes.at(i);
                    if (node.cluster == 0)
                        continue;

                    if (node.node == nodes.at(node.cluster))
                        node_outputs[i] = 1.0;
                }
            }
            else
            {
                // TODO: use subgradient data instead
                DualDecoder dual_decoder(status.n_cluster, status.arcs, status.nodes);
                dual_decoder.maximize(
                        status.cmsa_weights,
                        status.incoming_weights,
                        status.outgoing_weights,
                        status.node_weights,
                        [&] (const int i)
                        {
                            arc_outputs.at(i) += 1.0 / 3.0;
                        },
                        [&] (const int i)
                        {
                            arc_outputs.at(i) += 1.0 / 3.0;
                        },
                        [&] (const int i)
                        {
                            arc_outputs.at(i) += 1.0 / 3.0;
                        },
                        [&] (const int i)
                        {
                            node_outputs[i] = 1.0;
                        }
                );
            }

            n_total += sentence.size();

            // build loss output
            std::vector<Expression> errs;

            // for arcs
            for (unsigned i = 0 ; i < status.arcs.size() ; ++i)
            {
                auto const& arc = status.arcs[i];

                // if head is root, do not check the source_node, it will always be correct
                int head_pos = (arc.source > 0 ? sentence[arc.source].pos : arc.source_node);
                auto const& modifier = sentence[arc.destination];
                auto pred = arc_outputs[i];

                double gold = (head_pos == arc.source_node && modifier.pos == arc.destination_node && modifier.head == arc.source) ? 1.0 : 0.0;

                if (!NEARLY_EQ_TOL(pred, gold))
                {
                    auto expr = arc_exprs.at(i);

                    if (!NEARLY_ZERO_TOL(pred))
                    {
                        if (NEARLY_EQ_TOL(pred, 1.0))
                            errs.push_back(expr);
                        else
                            errs.push_back(pred * expr);
                    }
                    if (!NEARLY_ZERO_TOL(gold))
                    {
                        if (NEARLY_EQ_TOL(gold, 1.0))
                            errs.push_back(- expr);
                        else
                            errs.push_back(- gold * expr);
                    }
                }
                // equal to gold so we know it's binary
                else if (gold > 0.5)
                {
                    n_correct_head += 1.0;
                }
            }


            // for nodes
            for (unsigned i = 0 ; i < status.nodes.size() ; ++i)
            {
                auto const& node = status.nodes[i];

                // Do not check node if root
                if (node.cluster == 0)
                    continue;

                auto const& token_pos = sentence[node.cluster].pos;

                auto pred = node_outputs[i];

                double gold = (token_pos == node.node) ? 1.0 : 0.0;
                if (!NEARLY_EQ_TOL(pred, gold))
                {
                    auto expr = node_exprs.at(i);

                    if (!NEARLY_ZERO_TOL(pred))
                    {
                        if (NEARLY_EQ_TOL(pred, 1.0))
                            errs.push_back(expr);
                        else
                            errs.push_back(pred * expr);
                    }
                    if (!NEARLY_ZERO_TOL(gold))
                    {
                        if (NEARLY_EQ_TOL(gold, 1.0))
                            errs.push_back(- expr);
                        else
                            errs.push_back(- gold * expr);
                    }
                }
                // equal to gold so we know it's binary
                else if (gold > 0.5)
                {
                    n_correct_pos += 1;
                }
            }

            // backprop
            if (errs.size() > 0)
            {
                Expression sum_errs = dynet::expr::sum(errs);
                loss += as_scalar(cg.get_value(sum_errs.i));
                cg.backward(sum_errs.i);
                trainer.update(1.0);
            }
        }

        trainer.update_epoch();
        trainer.status();

        std::cerr 
            << "E = " 
            << (loss / (double) (n_total * 2)) 
            << " ppl=" 
            << exp(loss / (double) (n_total * 2)) 
            << std::endl
        ;
        std::cerr << "\tNode acc: " << n_correct_pos / (double) n_total << std::endl;
        std::cerr << "\tArc acc: " << n_correct_head / (double) n_total << std::endl;
        std::cerr << std::flush;

        save_object(model_path + ".param." + std::to_string(iteration), model);


        // Evaluate on dev set
        if (eval_on_dev && iteration >= start_eval_at)
        {
            unsigned n_exact = 0u;

            unsigned n_good_head = 0u;
            unsigned n_total_head = 0u;
            unsigned n_good_pos = 0u;
            unsigned n_total_pos = 0u;

            std::cerr << std::endl << "Evaluating on dev..." << std::endl << std::flush;

            for (auto const& sentence : dev_data)
            {
                std::vector<int> poss(sentence.size());
                std::vector<int> heads(sentence.size());
                DecoderTimer timer;

                bool exact = decode_primal(
                    sentence,
                    graph_generator,
                    stepsize_options,
                    max_iteration,
                    use_reduction,
                    rnn,
                    [&] (unsigned index, unsigned head)
                    {
                        heads[index] = head;
                    },
                    [&] (unsigned index, unsigned pos)
                    {
                        poss[index] = pos;
                    },
                    timer
                );

                if (exact)
                    ++ n_exact;

                for (unsigned i = 0u ; i < sentence.size() ; ++ i)
                {
                    if (sentence[i + 1].head == heads[i])
                        ++ n_good_head;
                    if (sentence[i + 1].pos == poss[i])
                        ++ n_good_pos;
                }
                n_total_head += sentence.size();
                n_total_pos += sentence.size();
            }

            std::cerr << " -> Converged: " << n_exact << " / " << dev_data.size() << std::endl;
            std::cerr << " -> UAS: " << n_good_head / (double) n_total_head << std::endl;
            std::cerr << " -> POS precision: " << n_good_pos / (double) n_total_pos << std::endl;
            std::cerr << " -> Overall precision: " << (n_good_pos + n_good_head) / (double) (n_total_pos + n_total_head) << std::endl << std::flush;
        }
        std::cerr << "------" << std::endl;
    }

    save_object(model_path + ".param", model);
}
