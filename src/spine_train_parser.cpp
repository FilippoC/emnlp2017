#include <iostream>
#include <vector>

#include <algorithm>
#include <unordered_map>
#include <set>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <exception>
#include <boost/filesystem.hpp>

#include "dynet/lstm.h"
#include "dynet/training.h"
#include "dynet/dynet.h"
#include "dynet/dict.h"

#include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/unordered_set.hpp>

#include "lemon_inc.h"
#include "serialization.h"
#include "nn/biaffine_parser.h"
#include "utils.h"

#include "dependency.h"
#include "spine_data.h"
#include "activation_function.h"


int main(int argc, char **argv)
{
    std::string train_path;
    std::string dev_path;
    std::string model_path;
    unsigned n_iteration;

    bool probabilistic;

    bool eval_on_dev;

    bool dropout;
    double dropout_p;

    std::string ignore_dynet_mem;
    std::string ignore_dynet_wd;

    unsigned n_stack;
    unsigned n_layer;
    int lstm_dim;
    int word_dim;
    int pos_dim;

    ActivationFunctionOption activation_function;
    bool best_head = false;

    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
        ("train", po::value<std::string>(&train_path)->required(), "")
        ("model", po::value<std::string>(&model_path)->required(), "")
        ("iteration", po::value<unsigned>(&n_iteration)->default_value(20), "")
        ("eval-on-dev", po::value<bool>(&eval_on_dev)->default_value(false), "")
        ("dev-path", po::value<std::string>(&dev_path)->default_value(""), "")
        // dropout
        ("dropout", po::value<bool>(&dropout)->default_value(false), "")
        ("dropout-p", po::value<double>(&dropout_p)->default_value(0.5), "")
        // dynet
        ("dynet-mem", po::value<std::string>(&ignore_dynet_mem), "")
        ("dynet-weight-decay", po::value<std::string>(&ignore_dynet_wd), "")
        // nn options
        ("activation-function", po::value<ActivationFunctionOption>(&activation_function), "") 
        ("lstm-dim", po::value<int>(&lstm_dim)->default_value(125))
        ("word-dim", po::value<int>(&word_dim)->default_value(100))
        ("pos-dim", po::value<int>(&pos_dim)->default_value(25))
        ("n-stack", po::value<unsigned>(&n_stack)->default_value(1u), "")
        ("n-layer", po::value<unsigned>(&n_layer)->default_value(1u), "")
        ("probabilistic", po::value<bool>(&probabilistic)->default_value(false), "")
        ("best-head", po::value<bool>(&best_head)->default_value(false))
    ;

    po::positional_options_description pod; 
    pod.add("train", 1); 
    pod.add("model", 1); 

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pod).run(), vm);
    po::notify(vm);

    if (probabilistic && !best_head)
        throw std::runtime_error("Use --best-head=true in order to use probabilistic network");

    dynet::initialize(argc, argv);

    SpineSettings spine_settings;
    read_object(model_path + ".spine_settings.param", spine_settings);
    
    std::cerr << "Reading train data..." << std::endl << std::flush;
    SpineData spine_train(spine_settings);
    std::vector<IntSentence> train_data;
    spine_train.read(train_path);
    spine_train.as_int_sentence([&](const IntSentence& s) { train_data.push_back(s); });


    std::vector<IntSentence> dev_data;
    if (eval_on_dev)
    {
        SpineData spine_dev(spine_settings);
        spine_dev.read(dev_path);
        spine_dev.as_int_sentence([&](const IntSentence& s) { dev_data.push_back(s); });
    }

    // TODO: other trainer
    dynet::Model model;
    dynet::AdamTrainer trainer(model);
    trainer.sparse_updates_enabled = false;

    RNNSettings node_rnn_settings;
    node_rnn_settings.padding = true;
    node_rnn_settings.dim = lstm_dim;
    node_rnn_settings.n_stack = n_stack;
    node_rnn_settings.n_layer = n_layer;


    NeuralBiaffineParserSettings nn_settings;
    nn_settings.n_word = spine_settings.word_dict.size();
    nn_settings.n_pos = spine_settings.pos_dict.size();
    nn_settings.word_unknown = spine_settings.word_dict.convert("*UNKNOWN*");
    nn_settings.word_dim = word_dim;
    nn_settings.pos_dim = pos_dim;
    nn_settings.pos_input = true;
    nn_settings.activation_function = activation_function.value;

    NeuralBiaffineParser<dynet::LSTMBuilder> rnn(model, nn_settings, node_rnn_settings);
    

    for (auto const& sentence : train_data)
    {
        // Count word frequence (used for dropout)
        for (auto const& token : sentence)
            rnn.m_word_count[token.word] += 1;
    }

    double loss = 0.0;

    save_object(model_path + ".parser.nn_settings", nn_settings);
    save_object(model_path + ".parser.rnn_settings", node_rnn_settings);

    std::vector<unsigned> dev_uas;

    for (unsigned iteration = 0 ; iteration <= n_iteration ; ++iteration)
    {
        std::cerr << "Iteration: " << iteration << std::endl;
        std::cerr << std::flush;

        unsigned n_correct = 0;
        unsigned n_total = 0;
        
        std::random_shuffle(std::begin(train_data), std::end(train_data));

        for (const IntSentence& sentence : train_data)
        {
            dynet::ComputationGraph cg;

            // loss output
            std::vector<Expression> errs;
            
            if (best_head)
            {
                std::vector<unsigned> correct_indices(sentence.size());
                std::vector<std::vector<Expression>> head_exprs(sentence.size());

                std::vector<std::vector<double>> scores(sentence.size());

                rnn.compute(
                    cg,
                    sentence,
                    [&] (unsigned head, unsigned modifier, double score, dynet::expr::Expression& expr) -> void
                    {
                        if ((int) head == sentence[modifier].head)
                            correct_indices.at(modifier-1) = head_exprs.at(modifier-1).size();
                        head_exprs.at(modifier-1).push_back(expr);

                        if (!probabilistic)
                        {
                            // loss augmented
                            if ((int) head == sentence[modifier].head)
                                score += 1.0;
                            
                            scores.at(modifier-1).push_back(score);
                        }
                    },
                    true, // word dropout
                    dropout,
                    dropout_p
                );

                if (probabilistic)
                {
                    for (unsigned i = 0u ; i < correct_indices.size() ; ++i)
                    {
                        // loss
                        dynet::expr::Expression heads = dynet::expr::concatenate(head_exprs.at(i));
                        errs.push_back(pickneglogsoftmax(heads, correct_indices.at(i)));

                        auto vec = dynet::as_vector(heads.value());
                        int predicted = std::distance(std::begin(vec), std::max_element(std::begin(vec), std::end(vec)));
                        if (predicted == (int) correct_indices.at(i))
                            n_correct += 1;
                    }
                }
                else
                {
                    for (unsigned i = 0u ; i < correct_indices.size() ; ++i)
                    {
                        // loss
                        dynet::expr::Expression heads = dynet::expr::concatenate(head_exprs.at(i));

                        auto vec = dynet::as_vector(heads.value());
                        int predicted = std::distance(std::begin(vec), std::max_element(std::begin(vec), std::end(vec)));
                        if (predicted == (int) correct_indices.at(i))
                        {
                            n_correct += 1;
                        }
                        else
                        {
                            errs.push_back(dynet::expr::pick(heads, predicted) - pick(heads, correct_indices.at(i)));
                        }
                    }
                }
            }
            else
            {
                // MSA decoding
                assert(!probabilistic);
                // TODO
                throw std::runtime_error("unimplemented");
            }
            n_total += sentence.size();

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

        std::cerr << "E = " << (loss / (double) n_total) << " ppl=" << exp(loss / (double) n_total) << " (acc=" << n_correct / (double) n_total << ")" << std::endl;
        std::cerr << std::flush;

        save_object(model_path + ".parser.param." + std::to_string(iteration), model);


        // Evaluate on dev set
        if (eval_on_dev)
        {
            unsigned n_good_head = 0u;
            unsigned n_total_head = 0u;

            std::cerr << "Evaluating on dev..." << std::endl << std::flush;

            for (auto const& sentence : dev_data)
            {
                dynet::ComputationGraph cg;

                LDigraph lemon_graph;
                LArcMap lemon_weights(lemon_graph);

                for (unsigned i = 0 ; i <= sentence.size() ; ++i)
                {
                    LNode node = lemon_graph.addNode();
                    assert(lemon_graph.id(node) == (int) i);
                }

                rnn.compute(
                    cg,
                    sentence,
                    [&] (unsigned head, unsigned modifier, double score, dynet::expr::Expression& expr) -> void
                    {
                        unused_parameter(expr);
                        LArc lemon_arc = lemon_graph.addArc(
                            lemon_graph.nodeFromId(head),
                            lemon_graph.nodeFromId(modifier)
                        );
                        lemon_weights[lemon_arc] = -score;
                    }
                );


                MSA msa(lemon_graph, lemon_weights);
                msa.run(lemon_graph.nodeFromId(0));

                for (unsigned modifier = 1 ; modifier <= sentence.size() ; ++modifier)
                {
                    auto msa_pred = msa.pred(lemon_graph.nodeFromId(modifier));
                    assert(msa_pred != lemon::INVALID);
                    
                    int predicted = lemon_graph.id(lemon_graph.source(msa_pred));
                    if (predicted == sentence[modifier].head)
                        ++ n_good_head;
                }


                n_total_head += sentence.size();
            }

            std::cerr << " -> UAS: " << n_good_head / (double) n_total_head << std::endl;
            dev_uas.push_back(n_good_head);
        }
        std::cerr << "------" << std::endl;
    }

    if (eval_on_dev)
    {
        unsigned max = std::distance(std::begin(dev_uas), std::max_element(std::begin(dev_uas), std::end(dev_uas)));
        boost::filesystem::copy_file(
                model_path + ".parser.param." + std::to_string(max),
                model_path + ".parser.param",
                boost::filesystem::copy_option::overwrite_if_exists
        );

    }
    else
    {
        // juste resave the last model
        save_object(model_path + ".parser.param", model);
    }
}

