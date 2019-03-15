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
#include "reduction.h"
#include "sgd.h"
#include "timer.h"
#include "writer.h"

#include "dependency.h"
#include "reader.h"


int main(int argc, char **argv)
{
    std::string test_path;
    std::string model_name;
    StepsizeOptions stepsize_options;
    bool verbose;
    unsigned max_iteration;
    bool use_reduction;
    bool use_cpos;

    bool with_stats;
    std::string stats_path;
    bool with_progress;

    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
        ("verbose", po::value<bool>(&verbose)->default_value(false), "")
        ("with-stats", po::value<bool>(&with_stats)->default_value(false), "")
        ("stats-path", po::value<std::string>(&stats_path)->default_value("stats.csv"), "")
        ("cpos", po::value<bool>(&use_cpos)->default_value(false), "")
        ("with-progress", po::value<bool>(&with_progress)->default_value(false), "")
        // Input files
        ("input", po::value<std::string>(&test_path)->required(), "Input file")
        ("model", po::value<std::string>(&model_name)->required(), "Model name")
        // SGD options
        ("reduction", po::value<bool>(&use_reduction)->default_value(false), "")
        ("max-iteration", po::value<unsigned>(&max_iteration)->default_value(500), "")
        ("stepsize-scale", po::value<double>(&stepsize_options.stepsize_scale)->default_value(1.0), "SGD: stepsize scale")
        ("polyak", po::value<bool>(&stepsize_options.polyak)->default_value(false), "SGD: use polyak steapsize")
        ("polyak-wub", po::value<double>(&stepsize_options.polyak_wub)->default_value(1.0), "SGD: weight of the UB (>= 1.0)")
        ("decreasing", po::value<bool>(&stepsize_options.decreasing)->default_value(true), "SGD: automatiquely decrease stepsize")
        ("constant-decreasing", po::value<bool>(&stepsize_options.constant_decreasing)->default_value(false), "SGD: decrease stepsize at each iteration")
        ("camerini", po::value<bool>(&stepsize_options.camerini)->default_value(false), "SGD: use Camerini et al. momentum subgradient")
        ("gamma", po::value<double>(&stepsize_options.gamma)->default_value(1.5), "SGD: gamma paremeter for Camerini et al. momentum subgradient")
    ;

    std::ofstream stats_fs;
    if (with_stats)
    {
        stats_fs.open(stats_path, std::ofstream::out);
        // TODO: output csv header
    }


    po::positional_options_description pod; 
    pod.add("input", 1); 
    pod.add("model", 1); 

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pod).run(), vm);
    po::notify(vm);

    dynet::initialize(argc, argv);

    dynet::Dict word_dict;
    dynet::Dict pos_dict;
    read_object(model_name + ".word_dict", word_dict);
    read_object(model_name + ".pos_dict", pos_dict);
    

    std::vector<IntSentence> test_data;

    std::cerr << "Reading test data..." << std::endl;
    read_conll_file(
            test_path,
            std::back_inserter(test_data),
            [&] (const StringToken& token) {
                return convert(word_dict, pos_dict, token);
            },
            use_cpos
    );

    std::cerr << "Nb sentences: " << test_data.size() << std::endl;


    dynet::Model model;

    RNNSettings shared_rnn_settings;
    RNNSettings arc_rnn_settings;
    RNNSettings node_rnn_settings;
    NNSettings nn_settings;
    NNArcSettings nn_arc_settings;
    NNNodeSettings nn_node_settings;

    read_object(model_name + ".nn_settings", nn_settings);
    read_object(model_name + ".shared_rnn_settings", shared_rnn_settings);
    read_object(model_name + ".arc_rnn_settings", arc_rnn_settings);
    read_object(model_name + ".node_rnn_settings", node_rnn_settings);
    read_object(model_name + ".arc_nn_settings", nn_arc_settings);
    read_object(model_name + ".node_nn_settings", nn_node_settings);

    NN<dynet::LSTMBuilder> rnn(model, nn_settings, shared_rnn_settings, arc_rnn_settings, node_rnn_settings, nn_arc_settings, nn_node_settings);
    
    read_object(model_name + ".param", model);
    
    // Allowed POS by word + allowed dependencies between POS
    GraphGenerator<IntSentence> graph_generator(pos_dict.size());
    read_object(model_name + ".graph_generator", graph_generator);

    // TODO: ugly wordaround
    // TODO2: I don't remember what's ugly here
    graph_generator.populate_with_everything(nn_settings.word_unknown);


    unsigned sentence_count = 0;
    for (auto const& sentence : test_data)
    {
        ++ sentence_count;

        if (with_progress)
            std::cerr << "\r" << sentence_count << "/" << test_data.size() << std::flush;

        std::vector<int> poss(sentence.size());
        std::vector<int> heads(sentence.size());

        decode(
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
            verbose
        );

        StringSentence out_sentence;
        for (unsigned i = 0u ; i < poss.size() ; ++ i)
        {
            out_sentence.tokens.emplace_back(
                    i + 1, // index
                    "UNK", // word
                    pos_dict.convert(poss[i]), // pos
                    heads[i] // head
            );
        }
        write_sentence(std::cout, out_sentence, use_cpos);
    }

    if (with_progress)
        std::cerr << std::endl;

    if (with_stats)
        stats_fs.close();
}

