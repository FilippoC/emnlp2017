#pragma once
#include <utility>
#include <unordered_map>
#include <boost/serialization/unordered_map.hpp>

struct SpineProbs
{
    std::unordered_map<int, double> root;
    std::unordered_map<std::pair<int, int>, double> non_root;

    std::unordered_map<std::pair<int, int>, std::pair<bool, unsigned>> attachments;

    template<class Archive> void serialize(Archive& ar, const unsigned int)
    {
        ar & root;
        ar & non_root;
        ar & attachments;
    }

};
