#pragma once

#include <iostream>
#include <fstream>
#include <regex>
#include <boost/algorithm/string.hpp>

#include "dependency.h"

struct ConllToken
{
    std::string id;
    std::string form;
    std::string lemma;
    std::string cpostag;
    std::string postag;
    std::string feats;
    std::string head;
    std::string deprel;
    std::string phead;
    std::string pdeprel;

    // TODO: overload >> to read an entire stream instead of this ugly workaround
    ConllToken(const std::string& line)
    {
        std::istringstream ss(line);
        ss >> id >> form >> lemma >> cpostag >> postag >> feats >> head >> deprel >> phead >> pdeprel;
    };
};

class ConllSentence : public std::vector<ConllToken>
{
};

struct ConllSettings
{
    bool to_num = true;
    bool to_lower = true;
    bool use_cpos = false;

    dynet::Dict word_dict;
    dynet::Dict pos_dict;

    void freeze()
    {
        word_dict.freeze();
        pos_dict.freeze();
    }

    template<class Archive> void serialize(Archive& ar, const unsigned int)
    {
        ar & to_num;
        ar & to_lower;
        ar & use_cpos;
        ar & word_dict;
        ar & pos_dict;
    }
};

class Conll : public std::vector<ConllSentence>
{
    std::regex num_regex;
    ConllSettings settings;

    public:

    Conll(ConllSettings& t_settings) : num_regex("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+"), settings(t_settings)
    {
    };

    void normalize(std::string& str)
    {
        // normalize word
        if (settings.to_num && std::regex_match(str, num_regex))
            str = "NUM";
        else if (settings.to_lower)
            boost::algorithm::to_lower(str);
    }

    template <typename OutputOp>
    void as_string_sentence(OutputOp op)
    {
        for (auto const& sentence : *this)
        {
            StringSentence str_sentence;
            for (auto const& token : sentence)
            {
                unsigned index = std::stoi(token.id);
                std::string word(token.form);
                std::string pos(settings.use_cpos ? token.cpostag : token.postag);
                unsigned head = std::stoi(token.head);

                normalize(word);

                str_sentence.push_back(StringToken(index, word, pos, head));
            }
            op(str_sentence);
        }
    }

    template <typename OutputOp>
    void as_int_sentence(OutputOp op)
    {
        for (auto const& sentence : *this)
        {
            IntSentence int_sentence;
            for (auto const& token : sentence)
            {
                unsigned index = std::stoi(token.id);
                std::string word(token.form);
                std::string pos(settings.use_cpos ? token.cpostag : token.postag);
                unsigned head = std::stoi(token.head);

                normalize(word);

                int_sentence.push_back(IntToken(
                            index, 
                            settings.word_dict.convert(word),
                            settings.pos_dict.convert(pos),
                            head
                ));
            }
            op(int_sentence);
        }
    }

    void read(const std::string& path)
    {
        clear();

        std::ifstream f(path);
        std::string line;

        ConllSentence* sentence = nullptr;

        while (std::getline(f, line)) {
            if (line.length() <= 0)
            {
                if (sentence != nullptr)
                {
                    push_back(*sentence);
                    delete sentence;
                    sentence = nullptr;
                }

                continue;
            }
            if (line[0] == '#')
                continue;

            if (sentence == nullptr)
                sentence = new ConllSentence();

            sentence->emplace_back(line);

        }

        if (sentence != nullptr)
        {
            push_back(*sentence);
            delete sentence;
        }
        f.close();
    }
};

std::ostream& operator<<(std::ostream& os, const ConllToken& token)  
{  

    os << token.id;
    os << "\t";
    os << token.form;
    os << "\t";
    os << token.lemma;
    os << "\t";
    os << token.cpostag;
    os << "\t";
    os << token.postag;
    os << "\t";
    os << token.feats;
    os << "\t";
    os << token.head;
    os << "\t";
    os << token.deprel;
    os << "\t";
    os << token.phead;
    os << "\t";
    os << token.pdeprel;

    return os;
}

std::ostream& operator<<(std::ostream& os, const ConllSentence& sentence)  
{  
    for (auto const& token : sentence)
        os << token << std::endl;

    return os;
}

std::ostream& operator<<(std::ostream& os, const Conll& conll)  
{  
    for (auto const& sentence : conll)
        os << sentence << std::endl;

    return os;
}
