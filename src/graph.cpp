#include "graph.h"

void Node::operator=(const Node& t_other)
{
    cluster = t_other.cluster; 
    node = t_other.node;
}

bool Arc::operator<(const Arc& t_other) const
{
    if (source != t_other.source)
        return source < t_other.source;
    if (source_node != t_other.source_node)
        return source_node < t_other.source_node;
    if (destination != t_other.destination)
        return destination < t_other.destination;

    return destination_node < t_other.destination_node;
}

bool Arc::operator==(const Arc& t_other) const
{
    return 
        (source == t_other.source) 
        && 
        (source_node == t_other.source_node)
        &&
        (destination == t_other.destination)
        &&
        (destination_node == t_other.destination_node)
    ;
}

bool Arc::operator!=(const Arc& t_other) const
{
    return !(t_other == *this);
}

void Arc::operator=(const Arc& t_other)
{
    source = t_other.source; 
    source_node = t_other.source_node;
    destination = t_other.destination;
    destination_node = t_other.destination_node;
}

std::ostream& operator<<(std::ostream& stream, const Arc& arc)
{
    stream << "(" << arc.source << ", " << arc.source_node << ", " << arc.destination << ", " << arc.destination_node << ")";
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const Node& node)
{
    stream << "(" << node.cluster << ", " << node.node << ")";
    return stream;
}
