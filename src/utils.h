#pragma once

#include <utility>
#include <boost/functional/hash.hpp>

// Stolen from AD3
#define TOL 1e-9
#define NEARLY_EQ_TOL(a,b) (((a)-(b))*((a)-(b))<=(TOL))                                                                                                                                                    
#define NEARLY_BINARY(a) (NEARLY_EQ_TOL((a),1.0) || NEARLY_EQ_TOL((a),0.0))
#define NEARLY_ZERO_TOL(a) (((a)<=(TOL)) && ((a)>=(-(TOL))))
#define STRICTLY_INF(a,b) (((a) < (b)) && !NEARLY_EQ_TOL((a), (b)))
#define STRICTLY_SUP(a,b) (((a) > (b)) && !NEARLY_EQ_TOL((a), (b)))


template<class T> void unused_parameter(const T&) { }

namespace std
{

template <>
struct hash<std::pair<unsigned, unsigned>>
{
    std::size_t operator()(const std::pair<unsigned, unsigned>& p) const
    {
        std::hash<unsigned> hu;

        std::size_t seed = 0;
        boost::hash_combine(seed, hu(p.first));
        boost::hash_combine(seed, hu(p.second));
        return seed;
    }
};

template <>
struct hash<std::pair<int, int>>
{
    std::size_t operator()(const std::pair<int, int>& p) const
    {
        std::hash<int> hu;

        std::size_t seed = 0;
        boost::hash_combine(seed, hu(p.first));
        boost::hash_combine(seed, hu(p.second));
        return seed;
    }
};

template <>
struct hash<std::pair<bool, unsigned>>
{
    std::size_t operator()(const std::pair<bool, unsigned>& p) const
    {
        std::hash<unsigned> hu;
        std::hash<bool> hb;

        std::size_t seed = 0;
        boost::hash_combine(seed, hb(p.first));
        boost::hash_combine(seed, hu(p.second));
        return seed;
    }
};

}


template <class T>
void swap(T** a, T** b)
{
    T* tmp = *b;
    *b = *a;
    *a = tmp;
}

double dot(const std::vector<double>& v1, const std::vector<double>& v2)
{
    double ret = 0.0;
    unsigned size = std::min(v1.size(), v2.size());

    for (unsigned i = 0u ; i < size ; ++i)
        ret += v1[i] * v2[i];

    return ret;
}
