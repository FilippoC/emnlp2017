#pragma once

struct PrimalDecoder
{
    Status& status;

    PrimalDecoder(Status& t_status)
        : status(t_status)
    {}

    bool update()
    {
        // TODO: check if the slected nodes have changed
        double new_weight = 0.0;

        LDigraph lemon_graph;
        LArcMap lemon_weights(lemon_graph);

        std::vector<LNode> lemon_nodes;
        for (unsigned i = 0 ; i < status.selected_nodes.size() ; ++i)
        {
            LNode node = lemon_graph.addNode();
            assert(lemon_graph.id(node) == (int) i);

            lemon_nodes.push_back(node);
            auto n = status.selected_nodes.at(i);
            new_weight += status.node_weights.at(n);

            //std::cout << "Cluster: " << i << "\t Node: " << status.nodes.at(n) << std::endl;
        }
        //std::cout << "pos score: " << new_weight << std::endl;


        // TODO: this vector seems useless to me...
        std::vector<LArc> lemon_arcs;
        std::vector<unsigned> arc_indices;
        for (unsigned i = 0 ; i < status.arcs.size() ; ++i)
        {
            auto const& arc = status.arcs.at(i);
            if (
                    status.nodes.at(status.selected_nodes.at(arc.source)).node != arc.source_node 
                    || 
                    status.nodes.at(status.selected_nodes.at(arc.destination)).node != arc.destination_node
            )
                continue;


            LArc lemon_arc = lemon_graph.addArc(
                lemon_nodes.at(arc.source),
                lemon_nodes.at(arc.destination)
            );
            arc_indices.push_back(i);

            lemon_weights[lemon_arc] = -status.original_weights.at(i);
            lemon_arcs.push_back(lemon_arc);
            //std::cout << arc << "\t" << status.original_weights.at(i) << std::endl;
        }

        MSA msa(lemon_graph, lemon_weights);
        msa.run(lemon_nodes.at(0));

        new_weight -= msa.arborescenceCost();
        //std::cout << "MSA: " << -msa.arborescenceCost() << std::endl;
        //std::cout << "Total: " << new_weight << std::endl;
        //throw std::runtime_error("");

        // did we manage to build a primal solution ?
        for (unsigned i = 1 ; i < status.selected_nodes.size() ; ++i)
        {
            auto msa_pred = msa.pred(lemon_graph.nodeFromId(i));
            if (msa_pred == lemon::INVALID)
            {
                return false;
            }
        }

        if (STRICTLY_SUP(new_weight, status.primal_weight))
        {
            status.primal_weight = new_weight;
            status.erase_primal_solution();

            for (unsigned i = 1 ; i < status.selected_nodes.size() ; ++i)
            {
                auto msa_pred = msa.pred(lemon_graph.nodeFromId(i));
                status.primal_arcs[arc_indices[lemon_graph.id(msa_pred)]] = true;
            }

            return true;
        }

        return false;
    }
};

