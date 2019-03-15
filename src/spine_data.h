#pragma once

#include <iostream>
#include <fstream>
#include <regex>
#include <map>
#include <set>
#include <boost/algorithm/string.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/map.hpp>

#include "dependency.h"

struct SpineToken
{
    std::string id;
    std::string form;
    std::string pos;
    std::string tpl;
    std::string head;
    std::string att_position;
    std::string att_type;

    // TODO: overload >> to read an entire stream instead of this ugly workaround
    SpineToken(const std::string& line)
    {
        std::istringstream ss(line);
        ss >> id >> form >> pos >> tpl >> head >> att_position >> att_type;
    };
};

class SpineSentence : public std::vector<SpineToken>
{
};

struct SpineSettings
{
    bool to_num = true;
    bool to_lower = true;
    std::map<int, std::set<int>> allowed_tpl;

    dynet::Dict word_dict;
    dynet::Dict pos_dict;
    dynet::Dict tpl_dict;

    void freeze()
    {
        word_dict.freeze();
        pos_dict.freeze();
        tpl_dict.freeze();
    }

    template<class Archive> void serialize(Archive& ar, const unsigned int)
    {
        ar & to_num;
        ar & to_lower;
        ar & word_dict;
        ar & pos_dict;
        ar & tpl_dict;
        ar & allowed_tpl;
    }
};

class SpineData : public std::vector<SpineSentence>
{
    std::regex num_regex;
    SpineSettings settings;

    public:

    SpineData(SpineSettings& t_settings) : num_regex("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+"), settings(t_settings)
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
                std::string pos(token.pos);
                std::string tpl(token.tpl);
                unsigned head = std::stoi(token.head);
                unsigned position = (unsigned) std::stoi(token.att_position);
                bool regular = (token.att_type == "r" ? true : false);

                normalize(word);

                str_sentence.push_back(StringToken(index, word, pos, head, tpl, regular, position));
            }
            op(str_sentence);
        }
    }

    template <typename OutputOp>
    void as_int_sentence(OutputOp op, bool c_tpl=true)
    {
        for (auto const& sentence : *this)
        {
            IntSentence int_sentence;
            for (auto const& token : sentence)
            {
                unsigned index = std::stoi(token.id);
                std::string word(token.form);
                std::string pos(token.pos);
                std::string tpl(token.tpl);
                unsigned head = std::stoi(token.head);
                unsigned position = (unsigned) std::stoi(token.att_position);
                bool regular = (token.att_type == "r" ? true : false);

                normalize(word);

                int_sentence.push_back(IntToken(
                            index, 
                            settings.word_dict.convert(word),
                            settings.pos_dict.convert(pos),
                            head,
                            (c_tpl ? settings.tpl_dict.convert(tpl) : 0),
                            regular,
                            position
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

        SpineSentence* sentence = nullptr;

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
                sentence = new SpineSentence();

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

std::ostream& operator<<(std::ostream& os, const SpineToken& token)  
{  

    os << token.id;
    os << "\t";
    os << token.form;
    os << "\t";
    os << token.pos;
    os << "\t";
    os << token.tpl;
    os << "\t";
    os << token.head;
    os << "\t";
    os << token.att_position;
    os << "\t";
    os << token.att_type;

    return os;
}

std::ostream& operator<<(std::ostream& os, const SpineSentence& sentence)  
{  
    for (auto const& token : sentence)
        os << token << std::endl;

    return os;
}

std::ostream& operator<<(std::ostream& os, const SpineData& conll)  
{  
    for (auto const& sentence : conll)
        os << sentence << std::endl;

    return os;
}
