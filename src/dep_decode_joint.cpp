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
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>

#include "lemon_inc.h"
#include "serialization.h"
#include "nn/tagger.h"
#include "nn/biaffine_parser.h"
#include "utils.h"
#include "sgd.h"
#include "status.h"
#include "graph.h"
#include "decoder.h"

#include "dependency.h"
#include "spine_data.h"
#include "activation_function.h"
#include "probs.h"


int main(int argc, char **argv)
{
    std::string test_path;
    std::string model_path;
    std::string output_path;

    StepsizeOptions stepsize_options;
    bool use_reduction;
    bool arc_weight_heuristic;
    unsigned max_iteration;
    double att_weight = 1.0;

    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
        ("test", po::value<std::string>(&test_path)->required(), "")
        ("model", po::value<std::string>(&model_path)->required(), "")
        ("output", po::value<std::string>(&output_path)->required(), "")
        ("reduction", po::value<bool>(&use_reduction)->default_value(false), "")
        ("arc-weight-heuristic", po::value<bool>(&arc_weight_heuristic)->default_value(false), "")
        ("att-weight", po::value<double>(&att_weight)->default_value(1.0), "")
        // SGD options
        ("max-iteration", po::value<unsigned>(&max_iteration)->default_value(500), "")
        ("stepsize-scale", po::value<double>(&stepsize_options.stepsize_scale)->default_value(1.0), "SGD: stepsize scale")
        ("polyak", po::value<bool>(&stepsize_options.polyak)->default_value(false), "SGD: use polyak steapsize")
        ("polyak-wub", po::value<double>(&stepsize_options.polyak_wub)->default_value(1.0), "SGD: weight of the UB (>= 1.0)")
        ("decreasing", po::value<bool>(&stepsize_options.decreasing)->default_value(true), "SGD: automatiquely decrease stepsize")
        ("constant-decreasing", po::value<bool>(&stepsize_options.constant_decreasing)->default_value(false), "SGD: decrease stepsize at each iteration")
        ("camerini", po::value<bool>(&stepsize_options.camerini)->default_value(false), "SGD: use Camerini et al. momentum subgradient")
        ("gamma", po::value<double>(&stepsize_options.gamma)->default_value(1.5), "SGD: gamma paremeter for Camerini et al. momentum subgradient")
    ;

    po::positional_options_description pod; 
    pod.add("test", 1); 
    pod.add("model", 1); 
    pod.add("output", 1); 

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pod).run(), vm);
    po::notify(vm);

    dynet::initialize(argc, argv);

    SpineSettings spine_settings;
    read_object(model_path + ".conll_settings.param", conll_settings);
    
    std::cerr << "Reading test data..." << std::endl << std::flush;
    Conll conll_test(conll_settings);
    std::vector<IntSentence> test_data;
    conll_test.read(test_path);
    conll_test.as_int_sentence([&](const IntSentence& s) { test_data.push_back(s); });
    
    // Neural Network
    dynet::Model tagger_model;

    RNNSettings tagger_rnn_settings;
    read_object(model_path + ".tagger.rnn_settings", tagger_rnn_settings);

    NeuralTaggerSettings tagger_nn_settings;
    read_object(model_path + ".tagger.nn_settings", tagger_nn_settings);

    NeuralTagger<dynet::LSTMBuilder> tagger_nn(tagger_model, tagger_nn_settings, tagger_rnn_settings);
    read_object(model_path + ".tagger.param", tagger_model);


    dynet::Model parser_model;

    RNNSettings parser_rnn_settings;
    read_object(model_path + ".parser.rnn_settings", parser_rnn_settings);

    NeuralBiaffineParserSettings nn_settings;
    read_object(model_path + ".parser.nn_settings", nn_settings);

    NeuralBiaffineParser<dynet::LSTMBuilder> parser_nn(parser_model, nn_settings, parser_rnn_settings);
    read_object(model_path + ".parser.param", parser_model);

    Probs attachment_probs;
    read_object(model_path + ".attachment-probs", attachment_probs);

    std::vector<std::set<int>> allowed_pos(conll_settings.word_dict.size());
    read_object(model_path + ".pos_filter", allowed_pos);

    auto word_unknown = conll_settings.word_dict.convert("*UNKNOWN*");
    for (unsigned i = 0u ; i < conll_settings.pos_dict.size() ; ++i)
        allowed_pos.at(word_unknown).insert(i);

    for (IntSentence& sentence : test_data)
    {
        Timer creation_timer;
        Timer solver_timer;

        creation_timer.start();
        Status status;

        status.n_cluster = sentence.size() + 1;

        // Compute node scores
        // root node
        status.nodes.emplace_back(0, 0);
        status.node_weights.push_back(0.0);

        {
            dynet::ComputationGraph cg;
            tagger_nn.compute(
                cg,
                sentence,
                [&] (unsigned index, dynet::expr::Expression& expr) -> void
                {
                    auto vec = dynet::as_vector(expr.value());
                    assert(vec.size() == conll_settings.pos_dict.size());
                    for (auto i : allowed_pos.at(sentence[index+1].word))
                    {
                        status.nodes.emplace_back(index+1, i);
                        status.node_weights.push_back(vec.at(i));
                    }
                    /*
                    for (unsigned i = 0u ; i < vec.size() ; ++i)
                    {
                        status.nodes.emplace_back(index+1, i);
                        status.node_weights.push_back(vec.at(i));
                    }
                    */
                }
            );
        }

        // Compute arc scores
        {
            dynet::ComputationGraph cg;
            parser_nn.compute(
                cg,
                sentence,
                [&] (const unsigned head, const unsigned modifier, const double score, dynet::expr::Expression& expr) -> void
                {
                    unused_parameter(expr);

                    if (head == 0u)
                    {
                        for (auto mod_pos : allowed_pos.at(sentence[modifier].word))
                        // for(int mod_pos = 0 ; mod_pos < (int) conll_settings.pos_dict.size() ; ++mod_pos)
                        {
                            double new_score = score;
                            auto f = attachment_probs.head.find(mod_pos);

                            // skip if not candidate for dependency
                            if (f == std::end(attachment_probs.head))
                                continue;

                            new_score += att_weight * log(f->second);

                            status.arcs.emplace_back(0, 0, modifier, mod_pos);
                            status.original_weights.push_back(new_score);
                        }
                    }
                    else
                    {
                        for (auto head_pos : allowed_pos.at(sentence[head].word))
                        //for (int head_pos = 0 ; head_pos < (int) conll_settings.pos_dict.size() ; ++head_pos)
                        {
                            for (auto mod_pos : allowed_pos.at(sentence[modifier].word))
                            //for (int mod_pos = 0 ; mod_pos < (int) conll_settings.pos_dict.size() ; ++mod_pos)
                            {
                                double new_score = score;
                                auto f = attachment_probs.pos.find(std::make_pair(head_pos, mod_pos));
                                // skip if not candidate for dependency
                                if (f == std::end(attachment_probs.pos))
                                    continue;

                                new_score += log(f->second);

                                status.arcs.emplace_back(head, head_pos, modifier, mod_pos);
                                status.original_weights.push_back(new_score);
                            }
                        }
                    }
                }
            );

            // heuristique for faster convergence
            if (arc_weight_heuristic)
            {
                std::vector<std::vector<double>> head_arc_weights(sentence.size() + 1);
                for (unsigned i = 0u ; i < status.arcs.size() ; ++i)
                {
                    auto const& arc = status.arcs.at(i);
                    double arc_weight = status.original_weights.at(i);

                    head_arc_weights.at(arc.destination).push_back(arc_weight);
                }

                std::vector<double> pad(sentence.size() + 1);
                for (unsigned i = 1u ; i < head_arc_weights.size() ; ++i)
                {
                    auto& weights = head_arc_weights.at(i);
                    if (weights.size() >= 2)
                    {
                        std::sort(std::begin(weights), std::end(weights), std::greater<double>());
                        pad.at(i) = - (weights.at(0) + weights.at(1)) / 2.0;
                    }
                    else
                    {
                        pad.at(i) = 0.0;
                    }
                }

                for (unsigned i = 0u ; i < status.arcs.size() ; ++i)
                {
                    auto const& arc = status.arcs.at(i);
                    status.original_weights.at(i) += pad.at(arc.destination);
                }
            }
        }
        creation_timer.stop();
        solver_timer.start();

        // decode
        DecoderTimer decoder_timer;
        decode_primal(
            status,
            stepsize_options,
            max_iteration,
            use_reduction,
            [&] (int index, int pos) {
                if (index != 0)
                    sentence[index].pos = pos;
            },
            [&] (int index, int head) {
                sentence[index].head = head;
            },
            decoder_timer
        );
        solver_timer.stop();

        //std::cout << creation_timer.milliseconds() << "\t" << solver_timer.milliseconds() << "\t" << sentence.size() << std::endl << std::flush;
        //std::cout << decoder_timer << std::endl;

        //std::cout << "Converged: " << converged << std::endl;
    } // end for sentence


    // Output
    for (unsigned i = 0u ; i < test_data.size() ; ++i)
    {
        const IntSentence& int_sentence = test_data.at(i);
        ConllSentence& conll_sentence = conll_test.at(i);

        for (unsigned j = 0u ; j < int_sentence.size() ; ++ j)
        {
            const auto& int_token = int_sentence[j+1];
            auto& conll_token = conll_sentence.at(j);

            conll_token.head = std::to_string(int_token.head);

            if (conll_settings.use_cpos)
            {
                conll_token.cpostag = conll_settings.pos_dict.convert(int_token.pos);
                conll_token.postag = "_";
            }
            else
            {
                conll_token.cpostag = "_";
                conll_token.postag = conll_settings.pos_dict.convert(int_token.pos);
            }
        }
    }

    std::ofstream f(output_path);
    f << conll_test;
    f.close();
}

