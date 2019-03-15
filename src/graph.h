#pragma once

#include <iostream>

class Node
{
public:
    int cluster;
    int node;

    Node(const int t_cluster, const int t_node)
        : cluster(t_cluster), node(t_node)
    {}

    void operator=(const Node& t_other);
};

class Arc 
{
    public:
        int source;
        int source_node;
        int destination;
        int destination_node;

        Arc(const int t_source, const int t_source_node, const int t_destination, const int t_destination_node)
            :
            source(t_source),
            source_node(t_source_node),
            destination(t_destination),
            destination_node(t_destination_node)
        {}

        bool operator<(const Arc& t_other) const;
        bool operator==(const Arc& t_other) const;
        bool operator!=(const Arc& t_other) const;
        void operator=(const Arc& t_other);

        inline bool is(int const t_source, int const t_source_node, int const t_destination, int const t_destination_node) const
        {
            return (source == t_source) && (source_node == t_source_node) && (destination == t_destination) && (destination_node == t_destination_node);
        }


        friend std::ostream& operator<<(std::ostream& stream, const Arc& arc);
};

std::ostream& operator<<(std::ostream& stream, const Arc& arc);
std::ostream& operator<<(std::ostream& stream, const Node& Node);
