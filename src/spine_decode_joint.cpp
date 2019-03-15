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
#include <boost/serialization/priority_queue.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>

#include "lemon_inc.h"
#include "serialization.h"
#include "nn/tagger.h"
#include "nn/head_tagger.h"
#include "nn/biaffine_parser.h"
#include "utils.h"
#include "sgd.h"
#include "status.h"
#include "graph.h"
#include "decoder.h"

#include "dependency.h"
#include "spine_data.h"
#include "activation_function.h"
#include "spine_probs.h"


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
    std::string unused;

    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
        ("test", po::value<std::string>(&test_path)->required(), "")
        ("model", po::value<std::string>(&model_path)->required(), "")
        ("output", po::value<std::string>(&output_path)->required(), "")
        ("reduction", po::value<bool>(&use_reduction)->default_value(false), "")
        ("arc-weight-heuristic", po::value<bool>(&arc_weight_heuristic)->default_value(false), "")
        ("att-weight", po::value<double>(&att_weight)->default_value(1.0), "")
        ("dynet-mem", po::value<std::string>(&unused)->default_value(""), "")
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
    read_object(model_path + ".spine_settings.param", spine_settings);
    
    std::cerr << "Reading test data..." << std::endl << std::flush;
    SpineData spine_test(spine_settings);
    std::vector<IntSentence> test_data;
    spine_test.read(test_path);
    spine_test.as_int_sentence([&](const IntSentence& s) { test_data.push_back(s); }, false);
    
    // Neural Network
    dynet::Model tagger_model;

    RNNSettings tagger_rnn_settings;
    read_object(model_path + ".tagger.rnn_settings", tagger_rnn_settings);

    NeuralTaggerSettings tagger_nn_settings;
    read_object(model_path + ".tagger.nn_settings", tagger_nn_settings);

    NeuralTagger<dynet::LSTMBuilder> tagger_nn(tagger_model, tagger_nn_settings, tagger_rnn_settings);
    read_object(model_path + ".tagger.param", tagger_model);

    dynet::Model head_tagger_model;

    RNNSettings head_tagger_rnn_settings;
    read_object(model_path + ".head_tagger.rnn_settings", head_tagger_rnn_settings);

    NeuralTaggerSettings head_tagger_nn_settings;
    read_object(model_path + ".head_tagger.nn_settings", head_tagger_nn_settings);

    NeuralTagger<dynet::LSTMBuilder> head_tagger_nn(head_tagger_model, head_tagger_nn_settings, head_tagger_rnn_settings);
    read_object(model_path + ".head_tagger.param", head_tagger_model);

    dynet::Model parser_model;

    RNNSettings parser_rnn_settings;
    read_object(model_path + ".parser.rnn_settings", parser_rnn_settings);

    NeuralBiaffineParserSettings nn_settings;
    read_object(model_path + ".parser.nn_settings", nn_settings);

    NeuralBiaffineParser<dynet::LSTMBuilder> parser_nn(parser_model, nn_settings, parser_rnn_settings);
    read_object(model_path + ".parser.param", parser_model);

    SpineProbs attachment_probs;
    read_object(model_path + ".attachment-probs", attachment_probs);

    std::vector<std::set<int>> allowed_spine(spine_settings.pos_dict.size());
    read_object(model_path + ".spine_filter", allowed_spine);

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

        std::vector<std::vector<int>> filters(sentence.size() + 1);
        {
            dynet::ComputationGraph cg;
            tagger_nn.compute(
                cg,
                sentence,
                [&] (unsigned index, dynet::expr::Expression& expr) -> void
                {
                    auto vec = dynet::as_vector(expr.value());
                    assert(vec.size() == spine_settings.tpl_dict.size());
                    /*
                    for (auto i : allowed_spine.at(sentence[index+1].pos))
                    {
                        status.nodes.emplace_back(index+1, i);
                        status.node_weights.push_back(vec.at(i));
                    }
                    */
					std::priority_queue<std::pair<double, int>> q;
					for (unsigned i = 0; i < vec.size(); ++i) {
						q.push(std::pair<double, int>(vec[i], i));
					}

					int k = 10;
					for (int i = 0; i < k; ++i) {
                        int ki = q.top().second;
                        status.nodes.emplace_back(index+1, ki);
                        status.node_weights.push_back(vec.at(ki));
                        filters.at(index+1).push_back(ki);
                        q.pop();
					}
				}
			);
        }

        std::vector<std::vector<float>> head_spine_weights;
        {
            dynet::ComputationGraph cg;
            head_tagger_nn.compute(
                cg,
                sentence,
                [&] (unsigned index, dynet::expr::Expression& expr) -> void
                {
                    assert(index == head_spine_weights.size());
                    head_spine_weights.push_back(dynet::as_vector(expr.value()));
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
                        //for (auto mod_spine : allowed_spine.at(sentence[modifier].pos))
                        for (auto mod_spine : filters.at(modifier))
                        {
                            double new_score = score;
                            auto f = attachment_probs.root.find(mod_spine);

                            // skip if not candidate for dependency
                            if (f == std::end(attachment_probs.root))
                                continue;

                            //new_score += att_weight * log(f->second);
                            new_score += head_spine_weights.at(modifier - 1).back();

                            status.arcs.emplace_back(0, 0, modifier, mod_spine);
                            status.original_weights.push_back(new_score);
                        }
                    }
                    else
                    {
                        //for (auto head_spine : allowed_spine.at(sentence[head].pos))
                        for (auto head_spine : filters.at(head))
                        {
                            //for (auto mod_spine : allowed_spine.at(sentence[modifier].pos))
                            for (auto mod_spine : filters.at(modifier))
                            {
                                double new_score = score;
                                auto f = attachment_probs.non_root.find(std::make_pair(head_spine, mod_spine));
                                // skip if not candidate for dependency
                                if (f == std::end(attachment_probs.non_root))
                                    continue;

                                //std::cout << head << ", " << modifier << "\t" << head_spine << ", " << mod_spine << "\t" << f->second << "\t" << log(f->second) << std::endl;
                                //new_score += log(f->second);
                                new_score += head_spine_weights.at(modifier - 1).at(head_spine);

                                status.arcs.emplace_back(head, head_spine, modifier, mod_spine);
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
            [&] (int index, int tpl) {
                if (index != 0)
                    sentence[index].tpl = tpl;
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
        SpineSentence& spine_sentence = spine_test.at(i);

        for (unsigned j = 0u ; j < int_sentence.size() ; ++ j)
        {
            const auto& int_token = int_sentence[j+1];
            auto& spine_token = spine_sentence.at(j);

            spine_token.head = std::to_string(int_token.head);
            spine_token.tpl = spine_settings.tpl_dict.convert(int_token.tpl);

            bool regular = false;
            unsigned position = true;
            if (int_token.head != 0)
            {
                regular = attachment_probs.attachments.at(std::make_pair(int_sentence[int_token.head].tpl, int_token.tpl)).first;
                position = attachment_probs.attachments.at(std::make_pair(int_sentence[int_token.head].tpl, int_token.tpl)).second;
            }
            spine_token.att_position = std::to_string(position);
            spine_token.att_type = (regular ? "r" : "s");
        }
    }

    std::ofstream f(output_path);
    f << spine_test;
    f.close();
}

