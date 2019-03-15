#pragma once

#include "status.h"
#include "decoder/dual.h"


// force an unselected node to be selected in the cluster subproblems
bool reduction_node(Status& status, DualDecoder& decoder, double dual_weight)
{
    bool reduced = false;

    for (auto const& cluster_decoder : decoder.cluster_decoders)
    {
        double max_cluster_weight = cluster_decoder._weights_cache[cluster_decoder._max_index_cache];

        for (unsigned i = 0u ; i < cluster_decoder._weights_cache.size() ; ++i)
        {
            if (i == cluster_decoder._max_index_cache || !status.allowed_nodes[cluster_decoder.decoders[i].node_index])
                continue;

            if (STRICTLY_INF(dual_weight - max_cluster_weight + cluster_decoder._weights_cache[i], status.primal_weight))
            {
                reduced = true;

                status.allowed_nodes[cluster_decoder.decoders[i].node_index] = false;
                status.node_weights[cluster_decoder.decoders[i].node_index] = -std::numeric_limits<double>::infinity();

                for (int index : cluster_decoder.decoders[i].incoming_indices)
                {
                    status.allowed_arcs.at(index) = false;

                    status.cmsa_weights[index] = -std::numeric_limits<double>::infinity();
                    status.incoming_weights[index] = -std::numeric_limits<double>::infinity();
                    status.outgoing_weights[index] = -std::numeric_limits<double>::infinity();
                }
                for (auto const& v : cluster_decoder.decoders[i].outgoing_indices)
                    for (int index : v)
                    {
                        status.allowed_arcs[index] = false;

                        status.cmsa_weights[index] = -std::numeric_limits<double>::infinity();
                        status.incoming_weights[index] = -std::numeric_limits<double>::infinity();
                        status.outgoing_weights[index] = -std::numeric_limits<double>::infinity();
                    }
            }
        }
    }

    return reduced;
}

// change the incoming arc of the selected node in a cluster
bool reduction_incoming(Status& status, DualDecoder& decoder, double dual_weight)
{
    bool reduced = false;

    for (unsigned c = 1u ; c < decoder.cluster_decoders.size(); ++ c)
    {
        auto const& cluster_decoder = decoder.cluster_decoders.at(c);
        unsigned i = cluster_decoder._max_index_cache;

        int selected_index = cluster_decoder.decoders.at(i)._incoming_selected;
        double selected_weight = status.incoming_weights.at(selected_index);

        for (unsigned index : cluster_decoder.decoders.at(i).incoming_indices)
        {
            if ((!status.allowed_arcs.at(index)) || index == (unsigned) cluster_decoder.decoders.at(i)._incoming_selected)
                continue;

            if (STRICTLY_INF(dual_weight - selected_weight + status.incoming_weights.at(index), status.primal_weight))
            {
                reduced = true;

                status.allowed_arcs.at(index) = false;

                status.cmsa_weights[index] = -std::numeric_limits<double>::infinity();
                status.incoming_weights[index] = -std::numeric_limits<double>::infinity();
                status.outgoing_weights[index] = -std::numeric_limits<double>::infinity();
            }
        }
    }

    return reduced;
}


// add an outgoing arc to the selected node in a cluster
bool reduction_outgoing_0(Status& status, DualDecoder& decoder, double dual_weight)
{
    bool reduced = false;

    for (auto const& cluster_decoder : decoder.cluster_decoders)
    {
        unsigned i = cluster_decoder._max_index_cache;
        for (unsigned j = 0 ; j < cluster_decoder.decoders.at(i)._outgoing_selected.size() ; ++j)
        {
            double selected_weight = 0.0;
            if (cluster_decoder.decoders.at(i)._outgoing_selected.at(j) >= 0)
                selected_weight = status.outgoing_weights.at(cluster_decoder.decoders.at(i)._outgoing_selected.at(j));

            for (unsigned index : cluster_decoder.decoders.at(i).outgoing_indices.at(j))
            {
                if ((!status.allowed_arcs.at(index)) || (cluster_decoder.decoders.at(i)._outgoing_selected.at(j) >= 0 && index == (unsigned) cluster_decoder.decoders.at(i)._outgoing_selected.at(j)))
                    continue;

                if (STRICTLY_INF(dual_weight - selected_weight + status.outgoing_weights.at(index), status.primal_weight))
                {
                    reduced = true;

                    status.allowed_arcs.at(index) = false;

                    status.cmsa_weights[index] = -std::numeric_limits<double>::infinity();
                    status.incoming_weights[index] = -std::numeric_limits<double>::infinity();
                    status.outgoing_weights[index] = -std::numeric_limits<double>::infinity();
                }
            }
        }
    }

    return reduced;
}


void remove_inaccessibles(Status& status, DualDecoder& decoder)
{
    bool has_changed;
    do
    {
        has_changed = false;
        for (unsigned c = 1u ; c < decoder.cluster_decoders.size() ; ++c)
        {
            auto const& cluster_decoder = decoder.cluster_decoders[c];

            for (auto const& node_decoder : cluster_decoder.decoders)
            {
                if (!status.allowed_nodes[node_decoder.node_index])
                    continue;

                bool accessible = false;
                for (int index : node_decoder.incoming_indices)
                {
                    if (status.allowed_arcs[index])
                    {
                        accessible = true;
                        break;
                    }
                }
                if (accessible)
                    continue;

                status.allowed_nodes[node_decoder.node_index] = false;
                status.node_weights[node_decoder.node_index] = -std::numeric_limits<double>::infinity();

                for (auto const& v : node_decoder.outgoing_indices)
                    for (int index : v)
                    {
                        has_changed = true;

                        status.allowed_arcs[index] = false;

                        status.cmsa_weights[index] = -std::numeric_limits<double>::infinity();
                        status.incoming_weights[index] = -std::numeric_limits<double>::infinity();
                        status.outgoing_weights[index] = -std::numeric_limits<double>::infinity();
                    }
            }
        }
    } while (has_changed);
}

bool reduction(Status& status, DualDecoder& decoder, double dual_weight)
{
    if (
        reduction_node(status, decoder, dual_weight)
        //|
        //reduction_incoming(status, decoder, dual_weight)
        //|
        //reduction_outgoing_0(status, decoder, dual_weight)
    )
    {
        //remove_inaccessibles(status, decoder);
        return true;
    }
    return false;
}
