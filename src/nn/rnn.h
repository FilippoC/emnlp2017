#pragma once

struct RNNSettings
{
    unsigned dim = 125;
    unsigned n_layer = 1;
    unsigned n_stack = 1;
    bool padding = false;

    template<class Archive> void serialize(Archive& ar, const unsigned int)
    {
        ar & dim;
        ar & n_layer;
        ar & n_stack;
        ar & padding;
    }
};

template <typename Builder>
struct RNN
{
    public:
        const RNNSettings settings;
        const unsigned input_dim;
        const int pad_begin = 0;
        const int pad_end = 1;

        dynet::LookupParameter pad;

        std::vector<Builder> builder_forwards;
        std::vector<Builder> builder_backwards;

    explicit RNN(dynet::Model &model, RNNSettings t_settings, unsigned t_input_dim)
        : settings(t_settings), input_dim(t_input_dim) 
    {
        builder_forwards.reserve(settings.n_stack);
        builder_backwards.reserve(settings.n_stack);

        for (unsigned i = 0 ; i < settings.n_stack ; ++ i)
        {
            if (i == 0)
            {
                builder_forwards.emplace_back(settings.n_layer, input_dim, settings.dim, model);
                builder_backwards.emplace_back(settings.n_layer, input_dim, settings.dim, model);
            }
            else
            {
                builder_forwards.emplace_back(settings.n_layer, output_dim(), settings.dim, model);
                builder_backwards.emplace_back(settings.n_layer, output_dim(), settings.dim, model);
            }
        }

        if (settings.padding)
        {
            pad = model.add_lookup_parameters(2, {input_dim});
        }
    }

    unsigned output_dim()
    {
        if (settings.n_stack == 0)
            return input_dim;
        else
            return settings.dim * 2;
    }

    void build(
        dynet::ComputationGraph& cg, 
        std::vector<dynet::expr::Expression>& input_embeddings,
        std::vector<dynet::expr::Expression>& output_embeddings,
        bool dropout=false,
        double dropout_p=0.5
    )
    {
        if (settings.n_stack == 0)
        {
            // just copy the input and quit
            output_embeddings = input_embeddings;
            return;
        }

        std::vector<dynet::expr::Expression> last_stack;
        for (unsigned stack = 0 ; stack < settings.n_stack ; ++stack)
        {
            std::vector<dynet::expr::Expression> lstm_forward;
            std::vector<dynet::expr::Expression> lstm_backward;

            lstm_forward.reserve(input_embeddings.size());
            lstm_backward.reserve(input_embeddings.size());

            // Forward
            auto& bf = builder_forwards.at(stack);
            if (dropout)
                bf.set_dropout(dropout_p);
            else
                bf.disable_dropout();

            bf.new_graph(cg);
            bf.start_new_sequence();

            if (settings.padding && stack == 0)
                bf.add_input(lookup(cg, pad, pad_begin));
            for (unsigned i = 0 ; i < input_embeddings.size() ; ++i)
            {
                if (stack == 0)
                    lstm_forward.push_back(bf.add_input(input_embeddings.at(i)));
                else
                    lstm_forward.push_back(bf.add_input(last_stack.at(i)));
            }
            if (settings.padding && stack == 0)
                bf.add_input(lookup(cg, pad, pad_end));

            // Backward
            auto& bb = builder_backwards.at(stack);
            if (dropout)
                bb.set_dropout(dropout_p);
            else
                bb.disable_dropout();

            bb.new_graph(cg);
            bb.start_new_sequence();
            if (settings.padding && stack == 0)
                bb.add_input(lookup(cg, pad, pad_end));
            for (int i = input_embeddings.size() - 1 ; i >= 0 ; --i)
            {
                if (stack == 0)
                    lstm_backward.push_back(bb.add_input(input_embeddings.at(i)));
                else
                    lstm_backward.push_back(bb.add_input(last_stack.at(i)));
            }
            if (settings.padding && stack == 0)
                bb.add_input(lookup(cg, pad, pad_begin));
            std::reverse(std::begin(lstm_backward), std::end(lstm_backward));


            // concatenate both lstms output
            // if last stack, copy to output
            auto& dest = (stack == settings.n_stack - 1 ? output_embeddings : last_stack);
            if (dest.size() > 0)
                dest.clear();
            dest.reserve(input_embeddings.size());

            for (unsigned i = 0 ; i < input_embeddings.size() ; ++i)
            {
                dest.push_back(
                    dynet::expr::concatenate({
                        lstm_forward.at(i),
                        lstm_backward.at(i)
                    })
                );
            }
        }
    }
};



