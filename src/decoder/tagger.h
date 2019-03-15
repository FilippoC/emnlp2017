#pragma once
#include "graph.h"


struct TaggerDecoder
{
    std::vector<std::vector<int>> tokens;

    TaggerDecoder(const int n_tokens, const std::vector<Node>& nodes)
        //: tokens(n_tokens)
    {
        for (int i = 0 ; i < n_tokens ; ++i)
            tokens.emplace_back();

        for (unsigned i = 0u ; i < nodes.size() ; ++i)
        {
            auto const& node = nodes.at(i);

            if (node.cluster == 0)
                continue;

            tokens.at(node.cluster - 1).push_back(i);
        }

        for (auto const v : tokens)
        {
            assert(v.size() > 0);
        }
    }


    template<class Op>
    double maximize(
        const std::vector<double>& node_weights,
        Op op
    )
    {
        double ret = 0.0;

        for (auto const& token : tokens)
        {
            int max_index = token.at(0u);
            double max_score = node_weights.at(max_index);

            for (unsigned i = 1u ; i < token.size() ; ++i)
            {
                int index = token.at(i);
                double score = node_weights.at(index);

                if (score > max_score)
                {
                    max_index = index;
                    max_score = score;
                }
            }

            op(max_index);
            ret += max_score;
        }

        return ret;
    }
};


