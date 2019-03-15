#include <iostream>
#include <vector>

#include "dynet/dict.h"
#include <set>

#include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
#include <boost/program_options.hpp>

#include "serialization.h"
#include "utils.h"
#include "conll.h"


int main(int argc, char **argv)
{
    std::string path;
    std::string model;

    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
        ("path", po::value<std::string>(&path)->required(), "")
        ("model", po::value<std::string>(&model)->required(), "")
    ;

    po::positional_options_description pod; 
    pod.add("path", 1); 
    pod.add("model", 1); 

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pod).run(), vm);
    po::notify(vm);

    ConllSettings conll_settings;
    read_object(model + ".conll_settings.param", conll_settings);
    
    Conll conll_train(conll_settings);
    std::vector<IntSentence> train_data;
    conll_train.read(path);
    conll_train.as_int_sentence([&](const IntSentence& s) { train_data.push_back(s); });

    std::vector<std::set<int>> allowed_pos(conll_settings.word_dict.size());
    auto unknown = conll_settings.word_dict.convert("*UNKNOWN*");
    for (auto const& sentence : train_data)
        for (auto const& token : sentence)
            if (token.word != unknown)
                allowed_pos.at(token.word).insert(token.pos);

    save_object(model + ".pos_filter", allowed_pos);

}

