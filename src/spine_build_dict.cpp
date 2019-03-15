#include <iostream>
#include <vector>

#include "dynet/dict.h"

#include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <boost/program_options.hpp>

#include "serialization.h"
#include "utils.h"
#include "spine_data.h"


int main(int argc, char **argv)
{
    std::string path;
    std::string output;
    unsigned word_threshold;

    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
        ("path", po::value<std::string>(&path)->required(), "")
        ("output", po::value<std::string>(&output)->required(), "")
        ("word-threshold", po::value<unsigned>(&word_threshold)->default_value(1u), "")
    ;

    po::positional_options_description pod; 
    pod.add("path", 1); 
    pod.add("output", 1); 

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pod).run(), vm);
    po::notify(vm);

    SpineSettings spine_settings;
    SpineData spine_data(spine_settings);
    spine_data.read(path);

    std::vector<StringSentence> sentences;
    spine_data.as_string_sentence([&] (const StringSentence& s) { sentences.push_back(s); });
   
    std::unordered_map<std::string, unsigned> counter;
    for (auto const& sentence : sentences)
        for (auto const&token : sentence)
            counter[token.word] += 1;

    for (auto const& word_c : counter)
        if (word_c.second >= word_threshold)
            spine_settings.word_dict.convert(word_c.first);

    for (auto const& sentence : sentences)
        for (auto const&token : sentence)
        {
            int pos = spine_settings.pos_dict.convert(token.pos);
            int tpl = spine_settings.tpl_dict.convert(token.tpl);
            spine_settings.allowed_tpl[pos].insert(tpl);
        }

    spine_settings.freeze();
    spine_settings.word_dict.set_unk("*UNKNOWN*");

    save_object(output + ".spine_settings.param", spine_settings);
}
