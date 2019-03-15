#pragma once

#include <utility>
#include <unordered_map>
#include <boost/serialization/unordered_map.hpp>

#include "utils.h"

struct Probs
{
    std::unordered_map<int, double> head;
    std::unordered_map<std::pair<int, int>, double> pos;

    template<class Archive> void serialize(Archive& ar, const unsigned int)
    {
        ar & head;
        ar & pos;
    }
};
