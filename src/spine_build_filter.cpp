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
#include "spine_data.h"


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

    SpineSettings spine_settings;
    read_object(model + ".spine_settings.param", spine_settings);
    
    SpineData spine_train(spine_settings);
    std::vector<IntSentence> train_data;
    spine_train.read(path);
    spine_train.as_int_sentence([&](const IntSentence& s) { train_data.push_back(s); });

    std::vector<std::set<int>> allowed_spine(spine_settings.tpl_dict.size());
    for (auto const& sentence : train_data)
        for (auto const& token : sentence)
                allowed_spine.at(token.tpl).insert(token.tpl);

    save_object(model + ".spine_filter", allowed_spine);

}

