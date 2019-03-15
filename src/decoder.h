#pragma once

#include <vector>
#include <assert.h>
#include <map>
#include <cstdlib>
#include <cmath>

#include "lemon_inc.h"
#include "graph.h"
#include "utils.h"
#include "status.h"
#include "dependency.h"
#include "graph_generator.h"
#include "timer.h"
#include "sgd.h"
#include "reduction.h"

#include "decoder/dual.h"
#include "decoder/primal.h"

struct DecoderTimer
{
    Timer total;
    Timer sgd;
    Timer solve_dual;
    Timer solve_primal;
    Timer sgd_update;
    Timer reduction;

    void stop()
    {
        total.stop(false);
        sgd.stop(false);
        solve_dual.stop(false);
        solve_primal.stop(false);
        sgd_update.stop(false);
        reduction.stop(false);
    }
};

std::ostream& operator<<(std::ostream& os, const DecoderTimer& t)  
{  
    os
        << t.total.milliseconds() << "\t"
        << t.sgd.milliseconds() << "\t"
        << t.solve_dual.milliseconds() << "\t"
        << t.solve_primal.milliseconds() << "\t"
        << t.sgd_update.milliseconds() << "\t"
        << t.reduction.milliseconds() << "\t"
    ;
    return os;
}

bool decode(
    Status& status,
    Subgradient& subgradient,
    unsigned max_iteration,
    bool use_reduction,
    DecoderTimer& timer,
    bool verbose=false
)
{
    timer.total.start();

    DualDecoder dual_decoder(status.n_cluster, status.arcs, status.nodes);
    PrimalDecoder primal_decoder(status);

    status.allowed_arcs.resize(status.arcs.size(), true);
    status.allowed_nodes.resize(status.nodes.size(), true);
    status.primal_arcs.resize(status.arcs.size(), false);

    status.selected_nodes.resize(status.n_cluster);


    remove_inaccessibles(status, dual_decoder);

    unsigned iteration = 0u;
    bool converged = false;


    timer.sgd.start();
    unsigned nb_wrong = 0u;
    for (; iteration < max_iteration; ++iteration)
    {
        subgradient.new_iteration();

        bool primal_change = (iteration == 0u ? true : false);

        timer.solve_dual.start();
        double dual_weight = dual_decoder.maximize(
                // TODO: decoder should save a reference to the status object
                status.cmsa_weights,
                status.incoming_weights,
                status.outgoing_weights,
                status.node_weights,
                [&] (const int i)
                {
                    assert(status.allowed_arcs[i]);
                    (*subgradient.gradient_cmsa)[i] += 1.0;
                },
                [&] (const int i)
                {
                    assert(status.allowed_arcs[i]);
                    (*subgradient.gradient_incoming)[i] += 1.0;
                },
                [&] (const int i)
                {
                    assert(status.allowed_arcs[i]);
                    (*subgradient.gradient_outgoing)[i] += 1.0;
                },
                [&] (const int i)
                {
                    assert(status.allowed_nodes.at(i));
                    auto const& node = status.nodes[i];
                    if (status.selected_nodes[node.cluster] != i)
                        primal_change = true;
                    status.selected_nodes[node.cluster] = i;
                }
        );
        timer.solve_dual.stop();

        // check for convergence
        converged = subgradient.is_null();

        if (converged)
        {
            status.primal_weight = dual_weight;
            status.dual_weight = dual_weight;

            // TODO: only construct it if the primal solution has changed !
            status.primal_from_subgradient(*subgradient.gradient_cmsa);

            break;
        }

        // check if dual value has increased
        if (iteration > 0 && dual_weight > status.dual_weight)
            subgradient.dual_has_increased();

        // TODO: check for primal change
        if (primal_change)
        {
            timer.solve_primal.start();
            primal_decoder.update();
            timer.solve_primal.stop();
        }

        // should be done in the decoder as for the primal
        status.dual_weight = std::min(status.dual_weight, dual_weight);

        // if equal, the primal is the optimal
        // (unlikely to happen while having a non null gradient, but we never know...)
        if (NEARLY_EQ_TOL(status.primal_weight, status.dual_weight))
        {
            converged = true;
            break;
        }


        unsigned nb_available_nodes = status.count_available_nodes();
        unsigned nb_available_arcs = status.count_available_arcs();
        // problem reduction
        if (use_reduction)
        {
            timer.reduction.start();
            reduction(status, dual_decoder, dual_weight);
            
            // Convergence test
            // if we have 1 node / cluster or n arcs left, we're done
            if (nb_available_nodes == status.n_cluster)
            {
                // set the nodes and then build the primal
                for (unsigned i = 0u ; i < status.nodes.size() ; ++i)
                {
                    if (status.allowed_nodes[i])
                    {
                        auto const& node = status.nodes[i];
                        status.selected_nodes[node.cluster] = i;
                    }
                }
                primal_decoder.update();

                converged = true;
                break;
            }
            if (nb_available_arcs == status.n_cluster - 1)
            {
                // TODO: pas sur que ça marche => on ne met pas à jour le score du primal
                status.primal_from_available_arcs();
                converged = true;
                break;
            }

            timer.reduction.stop();
        }


        // no need to update if it it the last iteration
        if (iteration < max_iteration - 1)
        {
            timer.sgd_update.start();
            nb_wrong = subgradient.update();
            timer.sgd_update.stop();
        }

        if (verbose)
        {
            std::cerr
                << iteration 
                << "\t" 
                << status.primal_weight 
                << "\t" 
                <<  status.dual_weight 
                << "\t" 
                << nb_wrong // will be invalid at the last iteration
                << "\t"
                << nb_available_nodes << "/" << status.allowed_nodes.size()
                << "\t"
                << nb_available_arcs << "/" << status.allowed_arcs.size()
                << std::endl;
        }
    }

    timer.stop();
    std::cout << converged << "\t" << iteration << std::endl << std::flush;


    return converged;
}

template <
    class SetPosOp,
    class SetHeadOp
>
bool decode_primal(
    Status& status,
    const StepsizeOptions& stepsize_options,
    unsigned max_iteration,
    bool use_reduction,
    SetPosOp set_pos_op,
    SetHeadOp set_head_op,
    DecoderTimer& timer,
    bool verbose=false
)
{

    // set submodel weights
    status.cmsa_weights.clear();
    status.incoming_weights.clear();
    status.outgoing_weights.clear();
    status.cmsa_weights.reserve(status.original_weights.size());
    status.incoming_weights.reserve(status.original_weights.size());
    status.outgoing_weights.reserve(status.original_weights.size());
    for (double weight : status.original_weights)
    {
        double w = weight / 3.0;
        status.cmsa_weights.push_back(w);
        status.incoming_weights.push_back(w);
        status.outgoing_weights.push_back(w);
    }

    Subgradient subgradient(stepsize_options, status);
    bool converged = decode(
        status,
        subgradient,
        max_iteration,
        use_reduction,
        timer,
        verbose
    );


    // do we have a finite primal solution ?
    if (std::isfinite(status.primal_weight))
    {
        double primal_score = 0.0;
        for (unsigned i = 0u ; i < status.arcs.size() ; ++i)
        {
            if (!status.primal_arcs[i])
                continue;

            auto const& arc = status.arcs[i];

            primal_score += status.original_weights[i];
            for (unsigned k = 0u ; k <  status.nodes.size() ; ++k)
            {
                auto const& node = status.nodes.at(k);
                if (node.cluster == arc.destination && node.node == arc.destination_node)
                    primal_score += status.node_weights.at(k);
            }

            set_pos_op(arc.destination, arc.destination_node);
            set_head_op(arc.destination, arc.source);

        }
        //std::cout << "primal_score: " << primal_score << std::endl;
    }
    else
    {
        // we return the last solution from the CMSA
        for (unsigned i = 0u ; i < subgradient.gradient_cmsa->size() ; ++i)
        {
            if (NEARLY_EQ_TOL((*subgradient.gradient_cmsa)[i], 0.0))
                continue;

            auto const& arc = status.arcs[i];

            set_pos_op(arc.destination, arc.destination_node);
            set_head_op(arc.destination, arc.source);
        }
    }


    return converged;
}


/*
template <
    class NNType,
    class SetArcOp,
    class SetNodeOp
>
bool decode_dual(
    const IntSentence& sentence, 
    const GraphGenerator<IntSentence>& graph_generator,
    const StepsizeOptions& stepsize_options,
    unsigned max_iteration,
    bool use_reduction,
    NNType& nn,
    SetArcOp set_arc_op,
    SetNodeOp set_node_op,
    DecoderTimer& timer,
    bool verbose=false
)
{
    timer.total.start();
    timer.graph_construction.start();

    dynet::ComputationGraph cg;

    Status status;
    status.n_cluster = sentence.size() + 1;

    graph_generator.build_arcs(
            sentence,
            [&] (const Arc& arc)
            {
                status.arcs.push_back(arc);
            },
            [&] (const Node& node)
            {
                status.nodes.push_back(node);
            }
    );


    nn.compute_exprs(
        cg,
        sentence,
        [&] (const Arc& arc, const double weight, dynet::expr::Expression& expr) -> void
        {
            unused_parameter(expr);
            unused_parameter(arc);

            status.original_weights.push_back(weight);

            double w = weight / 3.0;
            status.cmsa_weights.push_back(w);
            status.incoming_weights.push_back(w);
            status.outgoing_weights.push_back(w);
        },
        [&] (const Node node, const double weight, dynet::expr::Expression& expr) -> void
        {
            unused_parameter(expr);
            unused_parameter(node);

            status.node_weights.push_back(weight);
        },
        status.arcs,
        status.nodes,
        // dropout
        false
    );

    timer.graph_construction.stop();

    Subgradient subgradient(stepsize_options, status);
    bool converged = decode(
        status,
        subgradient,
        max_iteration,
        use_reduction,
        timer,
        verbose
    );


    // if converged primal = dual
    if (converged)
    {
        assert(std::isfinite(status.primal_weight));
        std::vector<int> nodes(status.n_cluster);

        for (unsigned i = 0u ; i < status.arcs.size() ; ++i)
        {
            if (!status.primal_arcs.at(i))
                continue;

            set_arc_op(i, 1.0);
            auto const arc = status.arcs.at(i);
            nodes.at(arc.destination) = arc.destination_node;
        }

        for (unsigned i = 0u ; i < status.nodes.size() ; ++i)
        {
            auto const node = status.nodes.at(i);
            if (node.cluster == 0)
                continue;

            if (node.node == nodes.at(node.cluster))
                set_node_op(i, 1.0);
        }
    }
    else
    {
        // TODO: use subgradient data instead
        std::vector<double> arcs(status.arcs.size(), 0.0);

        DualDecoder dual_decoder(status.n_cluster, status.arcs, status.nodes);
        double dual_weight = dual_decoder.maximize(
                status.cmsa_weights,
                status.incoming_weights,
                status.outgoing_weights,
                status.node_weights,
                [&] (const int i)
                {
                    arcs.at(i) += 1.0 / 3.0;
                },
                [&] (const int i)
                {
                    arcs.at(i) += 1.0 / 3.0;
                },
                [&] (const int i)
                {
                    arcs.at(i) += 1.0 / 3.0;
                },
                [&] (const int i)
                {
                    set_node_op(i, 1.0);
                }
        );

        for (unsigned i = 0u ; i < arcs.size() ; ++i)
        {
            if (STRICTLY_SUP(arcs.at(i), 0.0))
                set_arc_op(i, arcs.at(i));
        }
    }

    timer.total.stop();

    return converged;
}
*/
