#include <iostream>
#include <vector>

#include <unordered_map>
#include <exception>

#include "dynet/dict.h"

#include <boost/program_options.hpp>
#include <boost/serialization/unordered_set.hpp>

#include "serialization.h"
#include "utils.h"

#include "conll.h"
#include "dependency.h"
#include "spine_data.h"
#include "spine_probs.h"


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

    SpineSettings spine_settings;
    read_object(model_path + ".spine_settings.param", spine_settings);

    SpineData spine_train(spine_settings);
    std::vector<IntSentence> train_data;
    spine_train.read(data_path);
    spine_train.as_int_sentence([&](const IntSentence& s) { train_data.push_back(s); });


    std::vector<double> count_spines = std::vector<double>(spine_settings.tpl_dict.size(), 0.0);
    std::unordered_map<std::pair<int, int>, std::unordered_map<std::pair<bool, unsigned>, unsigned>> count_attachments;
    //std::unordered_map<std::pair<int, int>, unsigned> count_attachments;
    SpineProbs probs;
    for (auto const& sentence : train_data)
        for (auto const& token : sentence)
        {
            count_spines.at(token.tpl) += 1.0;

            if (token.head == 0)
                probs.root[token.tpl] += 1.0;
            else
            {
                probs.non_root[std::make_pair(sentence[token.head].tpl, token.tpl)] += 1.0;
                count_attachments[std::make_pair(sentence[token.head].tpl, token.tpl)][std::make_pair(token.regular, token.position)] += 1;
            }
        }

    for (auto const& it1 : count_attachments)
    {
        unsigned m = 0;
        for (auto const& it2 : it1.second)
        {
            if (it2.second > m)
            {
                probs.attachments[it1.first] = it2.first;
                m = it2.second;
            }
        }
    }

    std::for_each(
        std::begin(probs.root),
        std::end(probs.root),
        [&](std::pair<const int, double>& p) {
            p.second /= count_spines.at(p.first);
        }
    );
    std::for_each(
        std::begin(probs.non_root), 
        std::end(probs.non_root),
        [&](std::pair<const std::pair<int, int>, double>& p) {
            p.second /= count_spines.at(p.first.second);
        }
    );

    save_object(model_path + ".attachment-probs", probs);
}

