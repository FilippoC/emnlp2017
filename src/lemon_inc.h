#pragma once

#include <lemon/list_graph.h>
#include <lemon/concepts/maps.h>
#include <lemon/min_cost_arborescence.h>

typedef lemon::ListDigraph LDigraph;
typedef LDigraph::Node LNode;
typedef LDigraph::Arc LArc;
typedef double LWeight;
typedef LDigraph::ArcMap<LWeight> LArcMap;
typedef lemon::concepts::ReadWriteMap<LNode, LArc> LPredMap;
typedef lemon::concepts::WriteMap<LArc, bool> LArbMap;
typedef LDigraph::ArcIt LArcIt;
typedef lemon::MinCostArborescence<LDigraph, LArcMap> MSA;
