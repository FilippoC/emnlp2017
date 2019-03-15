#pragma once

#include "graph.h"

#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/unordered_set.hpp>


template<class SentenceType>
class GraphGenerator
{
    typedef typename SentenceType::WordType WordType;
    typedef typename SentenceType::POSType POSType;

    public:
        std::unordered_map<
            WordType,
            std::unordered_set<POSType>
        > m_allowed_pos_on_word;

        std::unordered_map<
            POSType,
            std::unordered_set<POSType>
        > m_allowed_head;

        std::unordered_map<
            std::pair<POSType, POSType>,
            int
        > max_left;

        std::unordered_map<
            std::pair<POSType, POSType>,
            int
        > max_right;

        unsigned m_nb_pos;
        unsigned m_root_pos;

    public:
        GraphGenerator(unsigned nb_pos)
            : m_nb_pos(nb_pos), m_root_pos(nb_pos)
        {}
            
        template<class Archive> void serialize(Archive& ar, const unsigned int)
        {
            ar & m_allowed_pos_on_word;
            ar & m_allowed_head;
            ar & max_left;
            ar & max_right;
            ar & m_nb_pos;
            ar & m_root_pos;
        }

        void update(const SentenceType& sentence)
        {
            for (auto const& token : sentence)
            {
                m_allowed_pos_on_word[token.word].insert(token.pos);
                if (token.head == 0)
                    m_allowed_head[token.pos].insert(m_root_pos);
                else
                {
                    m_allowed_head[token.pos].insert(sentence[token.head].pos);
                    auto key = std::make_pair(token.pos, sentence[token.head].pos);
                    if (token.head < token.index)
                        max_left[key] = std::max(max_left[key], token.index - sentence[token.head].index);

                    else
                        max_right[key] = std::max(max_right[key], sentence[token.head].index - token.index);
                }
            }
        }

        // TODO: this is an ugly workaround for something I forgot to do...
        void populate_with_everything(WordType word)
        {
            for (unsigned i = 0 ; i < m_nb_pos ; ++ i)
                m_allowed_pos_on_word[word].insert(i);
        }

        template <class ArcOp, class NodeOp>
        void build_arcs(const SentenceType& sentence, ArcOp arc_op, NodeOp node_op, bool limit=true)
        const
        {
            // add root node
            node_op(Node(0, m_root_pos));

            for (auto const& modifier : sentence)
            {
                for (auto const modifier_pos : m_allowed_pos_on_word.at(modifier.word))
                {
                    node_op(Node(modifier.index, modifier_pos));

                    auto const& allowed_heads = m_allowed_head.at(modifier_pos);
                    // add relation to root if allowed
                    if (allowed_heads.find(m_root_pos) != std::end(allowed_heads))
                    {
                        arc_op(Arc(
                                0,
                                m_root_pos,
                                modifier.index,
                                modifier_pos
                        ));
                    }

                    for (auto const& head : sentence)
                    {
                        if (modifier.index != head.index)
                        {
                            for (auto const head_pos : m_allowed_pos_on_word.at(head.word))
                            {
                                auto key = std::make_pair(modifier_pos, head_pos);

                                if (head.index < modifier.index)
                                {
                                    auto it = max_left.find(key);
                                    if (it == std::end(max_left))
                                        continue;
                                    if (limit && it->second < (modifier.index - head.index))
                                        continue;
                                }

                                if (modifier.index < head.index)
                                {
                                    auto it = max_right.find(key);
                                    if (it == std::end(max_right))
                                        continue;
                                    if (limit && it->second < (head.index - modifier.index))
                                        continue;
                                }

                                arc_op(Arc(
                                        head.index,
                                        head_pos,
                                        modifier.index,
                                        modifier_pos
                                ));
                            }
                        }
                    }
                }
            }
        }

};


