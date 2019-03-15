#include <iostream>
#include <vector>

#include "dynet/dict.h"

#include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <boost/program_options.hpp>

#include "serialization.h"
#include "utils.h"
#include "conll.h"


int main(int argc, char **argv)
{
    std::string path;
    std::string output;
    bool use_cpos;
    unsigned word_threshold;

    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
        ("path", po::value<std::string>(&path)->required(), "")
        ("output", po::value<std::string>(&output)->required(), "")
        ("cpos", po::value<bool>(&use_cpos)->default_value(false), "")
        ("word-threshold", po::value<unsigned>(&word_threshold)->default_value(1u), "")
    ;

    po::positional_options_description pod; 
    pod.add("path", 1); 
    pod.add("output", 1); 

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pod).run(), vm);
    po::notify(vm);

    ConllSettings conll_settings;
    conll_settings.use_cpos = use_cpos;
    Conll conll(conll_settings);
    conll.read(path);

    std::vector<StringSentence> sentences;
    conll.as_string_sentence([&] (const StringSentence& s) { sentences.push_back(s); });
   
    std::unordered_map<std::string, unsigned> counter;
    for (auto const& sentence : sentences)
        for (auto const&token : sentence)
            counter[token.word] += 1;

    for (auto const& word_c : counter)
        if (word_c.second >= word_threshold)
            conll_settings.word_dict.convert(word_c.first);

    for (auto const& sentence : sentences)
        for (auto const&token : sentence)
            conll_settings.pos_dict.convert(token.pos);

    conll_settings.freeze();
    conll_settings.word_dict.set_unk("*UNKNOWN*");

    save_object(output + ".conll_settings.param", conll_settings);
}
