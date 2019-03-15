#include <iostream>
#include <vector>

#include <algorithm>
#include <unordered_map>
#include <set>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <boost/filesystem.hpp>

#include "dynet/lstm.h"
#include "dynet/training.h"
#include "dynet/dynet.h"
#include "dynet/dict.h"

#include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/unordered_set.hpp>

#include "serialization.h"
#include "nn/tagger.h"
#include "utils.h"

#include "dependency.h"
#include "conll.h"
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

    ActivationFunctionOption activation_function;

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
        ("n-stack", po::value<unsigned>(&n_stack)->default_value(1u), "")
        ("n-layer", po::value<unsigned>(&n_layer)->default_value(1u), "")
        ("probabilistic", po::value<bool>(&probabilistic)->default_value(false), "")
    ;

    po::positional_options_description pod; 
    pod.add("train", 1); 
    pod.add("model", 1); 

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pod).run(), vm);
    po::notify(vm);

    dynet::initialize(argc, argv);

    ConllSettings conll_settings;
    read_object(model_path + ".conll_settings.param", conll_settings);
    
    std::cerr << "Reading train data..." << std::endl << std::flush;
    Conll conll_train(conll_settings);
    std::vector<IntSentence> train_data;
    conll_train.read(train_path);
    conll_train.as_int_sentence([&](const IntSentence& s) { train_data.push_back(s); });


    std::vector<IntSentence> dev_data;
    if (eval_on_dev)
    {
        Conll conll_dev(conll_settings);
        conll_dev.read(dev_path);
        conll_dev.as_int_sentence([&](const IntSentence& s) { dev_data.push_back(s); });
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


    NeuralTaggerSettings nn_settings;
    nn_settings.n_word = conll_settings.word_dict.size();
    nn_settings.n_pos = conll_settings.pos_dict.size();
    nn_settings.word_unknown = conll_settings.word_dict.convert("*UNKNOWN*");
    nn_settings.word_dim = word_dim;
    nn_settings.activation_function = activation_function.value;

    NeuralTagger<dynet::LSTMBuilder> rnn(model, nn_settings, node_rnn_settings);
    

    for (auto const& sentence : train_data)
    {
        // Count word frequence (used for dropout)
        for (auto const& token : sentence)
            rnn.m_word_count[token.word] += 1;
    }

    double loss = 0.0;

    save_object(model_path + ".tagger.nn_settings", nn_settings);
    save_object(model_path + ".tagger.rnn_settings", node_rnn_settings);

    std::vector<unsigned> dev_prec;

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
            
            rnn.compute(
                cg,
                sentence,
                [&] (unsigned index, dynet::expr::Expression& expr) -> void
                {
                    if (probabilistic)
                    {
                        errs.push_back(pickneglogsoftmax(expr, sentence[index+1].pos));

                        auto vec = dynet::as_vector(expr.value());
                        int predicted = std::distance(std::begin(vec), std::max_element(std::begin(vec), std::end(vec)));
                        if (predicted == sentence[index+1].pos)
                            n_correct += 1;
                    }
                    else
                    {
                        dynet::expr::Expression expr2 = tanh(expr);
                        auto vec = dynet::as_vector(expr2.value());

                        // loss augmented inference
                        for (unsigned i = 0u ; i < vec.size() ; ++i)
                        {
                            if ((int) i != sentence[index+1].pos)
                                vec[i] += 1.0;
                        }

                        int predicted = std::distance(std::begin(vec), std::max_element(std::begin(vec), std::end(vec)));
                        if (predicted == sentence[index+1].pos)
                        {
                            n_correct += 1;
                        }
                        else
                        {
                            errs.push_back(dynet::expr::pick(expr2, predicted) - pick(expr2, sentence[index+1].pos));
                        }
                    }
                },
                true, // word dropout
                dropout,
                dropout_p
            );

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

        save_object(model_path + ".tagger.param." + std::to_string(iteration), model);


        // Evaluate on dev set
        if (eval_on_dev)
        {
            unsigned n_good_pos = 0u;
            unsigned n_total_pos = 0u;

            std::cerr << "Evaluating on dev..." << std::endl << std::flush;

            for (auto const& sentence : dev_data)
            {
                dynet::ComputationGraph cg;

                rnn.compute(
                    cg,
                    sentence,
                    [&] (unsigned index, dynet::expr::Expression& expr) -> void
                    {
                        unused_parameter(expr);

                        if (probabilistic)
                        {
                            auto vec = dynet::as_vector(dynet::expr::softmax(expr).value());
                            int predicted = std::distance(std::begin(vec), std::max_element(std::begin(vec), std::end(vec)));
                            if (predicted == sentence[index+1].pos)
                                ++ n_good_pos;
                        }
                        else
                        {
                            auto vec = dynet::as_vector(dynet::expr::tanh(expr).value());
                            int predicted = std::distance(std::begin(vec), std::max_element(std::begin(vec), std::end(vec)));
                            if (predicted == sentence[index+1].pos)
                                ++ n_good_pos;
                        }
                    }
                );

                n_total_pos += sentence.size();
            }

            std::cerr << " -> POS precision: " << n_good_pos / (double) n_total_pos << std::endl;
            dev_prec.push_back(n_good_pos);
        }
        std::cerr << "------" << std::endl;
    }

    if (eval_on_dev)
    {
        unsigned max = std::distance(std::begin(dev_prec), std::max_element(std::begin(dev_prec), std::end(dev_prec)));
        boost::filesystem::copy_file(
                model_path + ".tagger.param." + std::to_string(max),
                model_path + ".tagger.param",
                boost::filesystem::copy_option::overwrite_if_exists
        );

    }
    else
    {
        // juste resave the last model
        save_object(model_path + ".tagger.param", model);
    }
}
