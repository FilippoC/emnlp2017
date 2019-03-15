#include "dependency.h"

IntToken convert(dynet::Dict& word_dict, dynet::Dict& pos_dict, const StringToken& t_token)
{
    int word = word_dict.convert(t_token.word);
    int pos = pos_dict.convert(t_token.pos);

    return IntToken(t_token.index, word, pos, t_token.head);
}

IntSentence convert(dynet::Dict& word_dict, dynet::Dict& pos_dict, const StringSentence& t_sentence)
{
    IntSentence sentence;
    std::transform(
        t_sentence.begin(),
        t_sentence.end(),
        sentence.back_inserter(),
        [&] (const StringToken& t_token)
        {
            return convert(word_dict, pos_dict, t_token);
        }
    );

    return sentence;
}
