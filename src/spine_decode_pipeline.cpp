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

    std::string unused;

    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
        ("test", po::value<std::string>(&test_path)->required(), "")
        ("model", po::value<std::string>(&model_path)->required(), "")
        ("output", po::value<std::string>(&output_path)->required(), "")
        ("dynet-mem", po::value<std::string>(&unused)->default_value(""), "")
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
            dynet::ComputationGraph cg;
            tagger_nn.compute(
                cg,
                sentence,
                [&] (unsigned index, dynet::expr::Expression& expr) -> void
                {
                    auto vec = dynet::as_vector(expr.value());
                    assert(vec.size() == spine_settings.tpl_dict.size());
                    int predicted = std::distance(std::begin(vec), std::max_element(std::begin(vec), std::end(vec)));
                    sentence[index+1].tpl = predicted;
				}
			);
    }

    for (IntSentence& sentence : test_data)
    {

        LDigraph lemon_graph;
        LArcMap lemon_weights(lemon_graph);

        std::vector<std::vector<float>> head_spine_weights;
        {
            for (unsigned i = 0 ; i <= sentence.size() ; ++i)
            {
                LNode node = lemon_graph.addNode();
                assert(lemon_graph.id(node) == (int) i);
            }

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

                    double new_score = score;
                    int mod_tpl = sentence[modifier].tpl;

                    if (head == 0u)
                    {
                            auto f = attachment_probs.root.find(mod_tpl);

                            // skip if not candidate for dependency
                            if (f == std::end(attachment_probs.root))
                                return;

                            //new_score += att_weight * log(f->second);
                            new_score += head_spine_weights.at(modifier - 1).back();
                    }
                    else
                    {
                        int head_tpl = sentence[head].tpl;
                        auto f = attachment_probs.non_root.find(std::make_pair(head_tpl, mod_tpl));
                        // skip if not candidate for dependency
                        if (f == std::end(attachment_probs.non_root))
                            return;

                        new_score += head_spine_weights.at(modifier - 1).at(head_tpl);
                    }
                    LArc lemon_arc = lemon_graph.addArc(
                        lemon_graph.nodeFromId(head),
                        lemon_graph.nodeFromId(modifier)
                    );
                    lemon_weights[lemon_arc] = -new_score;
                }
            );

        }
        MSA msa(lemon_graph, lemon_weights);
        msa.run(lemon_graph.nodeFromId(0));

        for (unsigned modifier = 1 ; modifier <= sentence.size() ; ++modifier)
        {
            auto msa_pred = msa.pred(lemon_graph.nodeFromId(modifier));
            assert(msa_pred != lemon::INVALID);
            
            int predicted = lemon_graph.id(lemon_graph.source(msa_pred));
            sentence[modifier].head = predicted;
        }
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


