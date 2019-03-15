#pragma once

#include <vector>

#include "graph.h"

struct Status
{
    unsigned n_cluster;

    double primal_weight;
    double dual_weight;

    std::vector<Arc> arcs;
    std::vector<Node> nodes;

    std::vector<bool> allowed_arcs;
    std::vector<bool> allowed_nodes;

    std::vector<bool> primal_arcs;

    std::vector<double> original_weights;
    std::vector<double> cmsa_weights;
    std::vector<double> incoming_weights;
    std::vector<double> outgoing_weights;
    std::vector<double> node_weights;

    std::vector<int> selected_nodes;

    Status()
    {
        primal_weight = -std::numeric_limits<double>::infinity();
        dual_weight = std::numeric_limits<double>::infinity();
    }

    void erase_primal_solution()
    {
        std::fill(
                std::begin(primal_arcs),
                std::end(primal_arcs),
                false
        );
    }

    void primal_from_available_arcs()
    {
        erase_primal_solution();
        for (unsigned i = 0u ; i < allowed_arcs.size() ; ++i)
        {
            if (allowed_arcs[i])
                primal_arcs[i] = true;
        }
    }

    void primal_from_subgradient(const std::vector<double>& v)
    {
        erase_primal_solution();
        for (unsigned i = 0u ; i < v.size() ; ++i)
        {
            if (NEARLY_EQ_TOL(v[i], 1.0))
                primal_arcs[i] = true;
        }
    }


    unsigned count_available_arcs()
    {
        unsigned nb_available_arcs = 0;
        for (bool b : allowed_arcs)
            if (b)
                ++ nb_available_arcs;
        return nb_available_arcs;
    }

    unsigned count_available_nodes()
    {
        unsigned nb_available_nodes = 0;
        for (bool b : allowed_nodes)
            if (b)
                ++ nb_available_nodes;
        return nb_available_nodes;
    }


};

