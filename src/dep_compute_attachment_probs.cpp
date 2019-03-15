#include <iostream>
#include <vector>

#include <unordered_map>
#include <exception>

#include "dynet/dict.h"

#include <boost/program_options.hpp>
#include <boost/serialization/unordered_set.hpp>

#include "serialization.h"
#include "utils.h"

#include "dependency.h"
#include "conll.h"
#include "probs.h"


int main(int argc, char **argv)
{
    std::string data_path;
    std::string model_path;

    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
        ("data", po::value<std::string>(&data_path)->required(), "")
        ("model", po::value<std::string>(&model_path)->required(), "")
    ;

    po::positional_options_description pod; 
    pod.add("data", 1); 
    pod.add("model", 1); 

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pod).run(), vm);
    po::notify(vm);

    ConllSettings conll_settings;
    read_object(model_path + ".conll_settings.param", conll_settings);
    
    std::cerr << "Reading data..." << std::endl << std::flush;
    Conll conll_data(conll_settings);
    std::vector<IntSentence> data;
    conll_data.read(data_path);
    conll_data.as_int_sentence([&](const IntSentence& s) { data.push_back(s); });

    Probs probs;
    std::vector<double> total(conll_settings.pos_dict.size(), 0.0);
    for (auto const& sentence : data)
    {
        for (auto const& token : sentence)
        {
            total.at(token.pos) += 1.0;

            if (token.head == 0)
                probs.head[token.pos] += 1.0;
            else
                probs.pos[std::make_pair(sentence[token.head].pos, token.pos)] += 1.0;
        }
    }
    std::for_each(
        std::begin(probs.head), 
        std::end(probs.head),
        [&](std::pair<const int, double>& p) {
            p.second /= total.at(p.first);
        }
    );
    std::for_each(
        std::begin(probs.pos), 
        std::end(probs.pos),
        [&](std::pair<const std::pair<int, int>, double>& p) {
            p.second /= total.at(p.first.second);
        }
    );

    save_object(model_path + ".attachment-probs", probs);
}
