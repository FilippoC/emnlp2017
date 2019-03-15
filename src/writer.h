#pragma once

#include <iostream>

#include "dependency.h"

template <class StreamType>
void write_sentence(StreamType& os, StringSentence& sentence, bool use_cpos=false)
{
    for (auto const& token : sentence)
    {
        os 
            << token.index << "\t"
            << token.word << "\t"
            << "_"<< "\t"
            << (use_cpos ? token.pos : "_")<< "\t"
            << (use_cpos ? "_" : token.pos )<< "\t"
            << "_"<< "\t"
            << token.head<< "\t"
            << "_"<< "\t"
            << "_"<< "\t"
            << "_"<< "\t"
            << std::endl;
        ;
    }
    os << std::endl;
}

