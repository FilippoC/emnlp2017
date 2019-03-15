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
#include "nn/tagger.h"
#include "nn/biaffine_parser.h"
#include "utils.h"

#include "dependency.h"
#include "conll.h"
#include "activation_function.h"
#include "probs.h"


int main(int argc, char **argv)
{
    std::string test_path;
    std::string model_path;
    std::string output_path;

    bool attachment_score;
    bool full;

    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
        ("test", po::value<std::string>(&test_path)->required(), "")
        ("model", po::value<std::string>(&model_path)->required(), "")
        ("output", po::value<std::string>(&output_path)->required(), "")
        ("attachment-score", po::value<bool>(&attachment_score)->default_value(true))
        ("full", po::value<bool>(&full)->default_value(false))
    ;

    po::positional_options_description pod; 
    pod.add("test", 1); 
    pod.add("model", 1); 
    pod.add("output", 1); 

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pod).run(), vm);
    po::notify(vm);

    dynet::initialize(argc, argv);

    ConllSettings conll_settings;
    read_object(model_path + ".conll_settings.param", conll_settings);
    
    std::cerr << "Reading test data..." << std::endl << std::flush;
    Conll conll_test(conll_settings);
    std::vector<IntSentence> test_data;
    conll_test.read(test_path);
    conll_test.as_int_sentence([&](const IntSentence& s) { test_data.push_back(s); });

    // Compute and update POS
    {
        // Neural Network
        dynet::Model model;

        RNNSettings node_rnn_settings;
        read_object(model_path + ".tagger.rnn_settings", node_rnn_settings);

        NeuralTaggerSettings nn_settings;
        read_object(model_path + ".tagger.nn_settings", nn_settings);

        NeuralTagger<dynet::LSTMBuilder> rnn(model, nn_settings, node_rnn_settings);

        read_object(model_path + ".tagger.param", model);


        for (IntSentence& sentence : test_data)
        {
            dynet::ComputationGraph cg;
            rnn.compute(
                cg,
                sentence,
                [&] (unsigned index, dynet::expr::Expression& expr) -> void
                {
                    auto vec = dynet::as_vector(expr.value());
                    int predicted = std::distance(std::begin(vec), std::max_element(std::begin(vec), std::end(vec)));
                    sentence[index+1].pos = predicted;
                }
            );
        }
    }


    // Compute and update dependencies
    {
        dynet::Model model;

        RNNSettings node_rnn_settings;
        read_object(model_path + ".parser.rnn_settings", node_rnn_settings);


        Probs attachment_probs;
        read_object(model_path + ".attachment-probs", attachment_probs);

        NeuralBiaffineParserSettings nn_settings;
        read_object(model_path + ".parser.nn_settings", nn_settings);

        NeuralBiaffineParser<dynet::LSTMBuilder> rnn(model, nn_settings, node_rnn_settings);
        read_object(model_path + ".parser.param", model);

        for (IntSentence& sentence : test_data)
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
                [&] (const unsigned head, const unsigned modifier, double score, dynet::expr::Expression& expr) -> void
                {
                    unused_parameter(expr);

                    int mod_pos = sentence[modifier].pos;


                    if (head == 0u)
                    {
                        auto f = attachment_probs.head.find(mod_pos);

                        if (!full)
                        {
                            // skip if not candidate for dependency
                            if (f == std::end(attachment_probs.head))
                                return;

                            if (attachment_score)
                                score += log(f->second);
                        }
                    }
                    else
                    {
                        int head_pos = sentence[head].pos;
                        auto f = attachment_probs.pos.find(std::make_pair(head_pos, mod_pos));

                        if (!full)
                        {
                            // skip if not candidate for dependency
                            if (f == std::end(attachment_probs.pos))
                                return;

                            if (attachment_score)
                                score += log(f->second);
                        }
                    }
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
                sentence[modifier].head = predicted;
            }
        }
    }


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
