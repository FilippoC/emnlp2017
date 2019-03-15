#pragma once

#include <vector>
#include <algorithm>
#include "dynet/dict.h"


template <typename TWordType, typename TPOSType>
struct Token
{
    public:
    typedef TWordType WordType;
    typedef TPOSType POSType;
    typedef TPOSType TplType;

    int index;
    WordType word;
    POSType pos;
    int head;
    TplType tpl;
    bool regular;
    unsigned position;

    Token(
        int t_index, 
        WordType t_word,
        POSType t_pos,
        int t_head
    )
    :
        index(t_index),
        word(t_word),
        pos(t_pos),
        head(t_head)
    {}

    Token(
        int t_index, 
        WordType t_word,
        POSType t_pos,
        int t_head,
        TplType t_tpl,
        bool t_regular,
        unsigned t_position
    )
    :
        index(t_index),
        word(t_word),
        pos(t_pos),
        head(t_head),
        tpl(t_tpl),
        regular(t_regular),
        position(t_position)
    {}
};

template <typename TokenType>
struct Sentence
{
    public:
        typedef typename std::vector<TokenType> TokenList;
        typedef typename TokenList::const_iterator const_iterator;

        typedef typename TokenType::WordType WordType;
        typedef typename TokenType::POSType POSType;

        TokenList tokens;

    public:
        unsigned int size() const
        {
            return tokens.size();
        }

        const TokenType& operator [](int i) const {
            assert(i > 0);
            assert(i <= (int) tokens.size());
            return tokens.at(i - 1);
        }

        TokenType& operator [](int i) {
            assert(i > 0);
            assert(i <= (int) tokens.size());
            return tokens.at(i - 1);
        }

        const_iterator begin() const
        {
            return std::begin(tokens);
        }

        const_iterator end() const
        {
            return std::end(tokens);
        }

        // TODO: the back insert should check that token identifiers are valid
        inline std::back_insert_iterator<TokenList> back_inserter()
        {
            return std::back_inserter(tokens);
        }

        void push_back(const TokenType& token)
        {
            tokens.push_back(token);
        }

        void push_back(TokenType&& token)
        {
            tokens.push_back(token);
        }
};

template<typename WordType, typename POSType>
std::ostream& operator<<(std::ostream& ostream, const Token<WordType, POSType>& token)
{
    ostream << token.index << "\t";
    ostream << token.word << "\t";
    ostream << token.pos << "\t";
    ostream << token.head;

    return ostream;
}

template<typename TokenType>
std::ostream& operator<<(std::ostream& ostream, const Sentence<TokenType>& sentence)
{
    for (auto const& token : sentence.tokens)
        ostream << token << std::endl;

    return ostream;
}

typedef Token<std::string, std::string> StringToken;
typedef Sentence<StringToken> StringSentence;

typedef Token<int, int> IntToken;
typedef Sentence<IntToken> IntSentence;

IntToken convert(dynet::Dict& word_dict, dynet::Dict& pos_dict, const StringToken& t_token);
IntSentence convert(dynet::Dict& word_dict, dynet::Dict& pos_dict, const StringSentence& t_sentence);

// TODO: remove this ? seems useless
template <class InputIt, class OutputIt>
OutputIt convert(dynet::Dict& word_dict, dynet::Dict& pos_dict, InputIt first, InputIt last, OutputIt d_first)
{
    return std::transform(
        first,
        last,
        d_first,
        [&] (const StringSentence& t_sentence)
        {
            return convert(word_dict, pos_dict, t_sentence);
        }
    );
}
