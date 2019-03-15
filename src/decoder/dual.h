#pragma once

struct CMSADecoder
{
    int cluster_size;

    LDigraph lemon_graph;
    LArcMap arc_weights;

    std::vector<std::vector<unsigned>> lemon_arcs;

    // used to build solution & for problem reduction
    std::vector<unsigned> _arc_cache;

    CMSADecoder(unsigned t_cluster_size, const std::vector<Arc>& arcs)
        : cluster_size(t_cluster_size), arc_weights(lemon_graph)
    {
        for (int i = 0 ; i < cluster_size; ++i)
        {
            auto node = lemon_graph.addNode();
            assert(lemon_graph.id(node) == i);
        }

        std::map<std::pair<int, int>, int> lemon_arcs_eq;
        for (unsigned i = 0 ; i < arcs.size() ; ++ i)
        {
            auto const& arc = arcs[i];
            auto carc = std::make_pair(arc.source, arc.destination);
            auto it = lemon_arcs_eq.find(carc);

            if (it == std::end(lemon_arcs_eq))
            {
                auto lemon_arc = lemon_graph.addArc(
                        lemon_graph.nodeFromId(arc.source), 
                        lemon_graph.nodeFromId(arc.destination)
                );
                int id = lemon_graph.id(lemon_arc);
                assert(id == (int) lemon_arcs.size());

                lemon_arcs_eq[carc] = id;
                lemon_arcs.emplace_back(1, i);
            }
            else
            {
                assert(it->second < (int) lemon_arcs.size());
                lemon_arcs[it->second].push_back(i);
            }
        }

        _arc_cache.resize(lemon_arcs.size());
    }

    template<class Operator>
    double maximize(const std::vector<double>& weights, Operator op)
    {
        for (int lemon_id = 0 ; lemon_id < (int) lemon_arcs.size() ; ++ lemon_id)
        {
            auto const& indices = lemon_arcs[lemon_id];
            
            unsigned max_index = indices[0u];
            double max_weight = weights[max_index];

            for (unsigned i = 1u ; i < indices.size() ; ++i)
            {
                double w = weights.at(indices.at(i));
                if (w > max_weight)
                {
                    max_index = indices[i];
                    max_weight = w;
                }
            }

            arc_weights[lemon_graph.arcFromId(lemon_id)] = -max_weight;
            _arc_cache[lemon_id] = max_index;
        }


        MSA msa(lemon_graph, arc_weights);
        msa.run(lemon_graph.nodeFromId(0));
        
        assert(std::isfinite(msa.arborescenceCost()));

        for (int i = 1 ; i < cluster_size ; ++i)
        {
            auto msa_pred = msa.pred(lemon_graph.nodeFromId(i));
            if (msa_pred == lemon::INVALID)
            {
                throw std::runtime_error("Failed to produce arborescence");
            }


            op(_arc_cache[lemon_graph.id(msa_pred)]);
        }

        return -msa.arborescenceCost();
    }
};

struct NodeDecoder
{
    std::vector<int> incoming_indices;
    std::vector<std::vector<int>> outgoing_indices;
    int node_index;

    // used for problem reduction
    // TODO: also use this instead of recomputing the max
    // when computing the solution
    std::vector<int> _outgoing_selected;
    int _incoming_selected;

    NodeDecoder(const int t_node_index, const int t_cluster_size)
        : outgoing_indices(t_cluster_size), node_index(t_node_index), _outgoing_selected(t_cluster_size)
    {};

    template<class Op1, class Op2, class Op3>
    double maximize(
        const std::vector<double>& incoming_weights, 
        const std::vector<double>& outgoing_weights,
        const std::vector<double>& node_weights,
        Op1 op_incoming,
        Op2 op_outgoing,
        Op3 op_node
    ) 
    {
        double total_weight = node_weights[node_index];
        op_node(node_index);

        _incoming_selected = -1;
        if (incoming_indices.size() > 0)
        {
            int max_index = incoming_indices[0u];
            double max_weight = incoming_weights[max_index];

            for (unsigned i = 1u ; i < incoming_indices.size() ; ++ i)
            {
                unsigned index = incoming_indices[i];
                double weight = incoming_weights[index];
                if (weight > max_weight)
                {
                    max_weight = weight;
                    max_index = index;
                }
            }
            total_weight += max_weight;
            _incoming_selected = max_index;
            op_incoming(max_index);
        }

        for (unsigned cluster_index = 0u ; cluster_index < outgoing_indices.size() ; ++ cluster_index)
        {
            auto const& indices = outgoing_indices[cluster_index];
            _outgoing_selected[cluster_index] = -1;

            if (indices.size() > 0)
            {
                unsigned max_index = 0u;
                double max_weight = -1.0;

                for (unsigned i = 0u ; i < indices.size() ; ++ i)
                {
                    int index = indices[i];
                    double weight = outgoing_weights[index];
                    if (weight > max_weight)
                    {
                        max_weight = weight;
                        max_index = index;
                    }
                }

                if (max_weight > 0.0)
                {
                    total_weight += max_weight;
                    op_outgoing(max_index);
                    _outgoing_selected[cluster_index] = max_index;
                }
            }
        }

        return total_weight;
    }
};

struct ClusterDecoder
{
    std::vector<NodeDecoder> decoders;
    int index;
    int cluster_size;

    // used for problem reduction
    std::vector<double> _weights_cache;
    unsigned _max_index_cache;

    ClusterDecoder(const int t_index, const int t_cluster_size, const std::vector<Arc>& arcs, const std::vector<Node> nodes)
        : index(t_index), cluster_size(t_cluster_size)
    {
        std::map<int, int> node_indices;
        for (unsigned i = 0u ; i < nodes.size() ; ++i)
        {
            auto const& node = nodes.at(i);
            if (node.cluster == index)
            {
                node_indices[node.node] = decoders.size();
                decoders.emplace_back(i, cluster_size);
            }
        }
        _weights_cache.resize(decoders.size());

        for (unsigned i = 0u ; i < arcs.size() ; ++i)
        {
            auto const& arc = arcs.at(i);

            // incoming arc
            if (arc.destination == index)
            {
                auto const& tmp = node_indices.at(arc.destination_node);
                decoders.at(tmp).incoming_indices.push_back(i);
                //decoders.at(node_indices.at(arc.destination_node)).incoming_indices.push_back(i);
            }

            // outgoing arc
            if (arc.source == index)
                decoders.at(node_indices.at(arc.source_node)).outgoing_indices.at(arc.destination).push_back(i);
        }
    }


    template<class Op1, class Op2, class Op3>
    double maximize(
        const std::vector<double>& incoming_weights, 
        const std::vector<double>& outgoing_weights,
        const std::vector<double>& node_weights,
        Op1 op_incoming,
        Op2 op_outgoing,
        Op3 op_node
    ) 
    {
        unsigned max_index = 0u;
        double max_weight = decoders[0u].maximize(
                incoming_weights,
                outgoing_weights,
                node_weights,
                [] (int i) { unused_parameter(i); },
                [] (int i) { unused_parameter(i); },
                [] (int i) { unused_parameter(i); }
        );
        _weights_cache[0u] = max_weight;

        for (unsigned i = 1u ; i < decoders.size() ; ++i)
        {
            double weight = decoders[i].maximize(
                    incoming_weights,
                    outgoing_weights,
                    node_weights,
                    [] (int i) { unused_parameter(i); },
                    [] (int i) { unused_parameter(i); },
                    [] (int i) { unused_parameter(i); }
            );
            _weights_cache[i] = weight;

            if (weight > max_weight)
            {
                max_weight = weight;
                max_index = i;
            }
        }

        _max_index_cache = max_index;

        // We need to recall the max because of the lambdas
        return decoders[max_index].maximize(
                incoming_weights,
                outgoing_weights,
                node_weights,
                op_incoming,
                op_outgoing,
                op_node
        );
    }
};
struct DualDecoder
{
    int cluster_size;
    CMSADecoder cmsa_decoder;
    std::vector<ClusterDecoder> cluster_decoders;

    DualDecoder(const int t_cluster_size, const std::vector<Arc>& arcs, const std::vector<Node>& nodes)
        : cluster_size(t_cluster_size), 
          cmsa_decoder(t_cluster_size, arcs)
    {
        for (int i = 0 ; i < cluster_size ; ++i)
            //cluster_decoders.emplace_back(i, cluster_size, arcs, nodes);
            cluster_decoders.push_back(ClusterDecoder(i, cluster_size, arcs, nodes));
    }

    template<class Op1, class Op2, class Op3, class Op4>
    double maximize(
        const std::vector<double>& cmsa_weight,
        const std::vector<double>& incoming_weights, 
        const std::vector<double>& outgoing_weights,
        const std::vector<double>& node_weights,
        Op1 op_cmsa,
        Op2 op_incoming,
        Op3 op_outgoing,
        Op4 op_node
    )
    {
        double weight = cmsa_decoder.maximize(cmsa_weight, op_cmsa);

        for (auto& decoder : cluster_decoders)
            weight += decoder.maximize(
                    incoming_weights,
                    outgoing_weights,
                    node_weights,
                    op_incoming,
                    op_outgoing,
                    op_node
            );

        return weight;
    }
};

