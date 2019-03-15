#pragma once

#include "status.h"
#include "utils.h"

struct StepsizeOptions
{
    double stepsize_scale = 1.0;
    bool camerini = false;
    bool polyak = false;
    double polyak_wub = 1.0;
    bool decreasing = true;
    bool constant_decreasing = false;
    double gamma = 1.5;
};

struct Subgradient
{
    typedef std::vector<double> GradientType;
    const StepsizeOptions options;
    Status& status;

    double gradient_norm;
    GradientType *gradient_cmsa;
    GradientType *gradient_incoming;
    GradientType *gradient_outgoing;

    double previous_gradient_norm;
    GradientType *previous_gradient_cmsa;
    GradientType *previous_gradient_incoming;
    GradientType *previous_gradient_outgoing;

    double iteration;
    double n_increasing;

    Subgradient(const StepsizeOptions& t_options, Status& t_status)
        : options(t_options), status(t_status)
    {
        iteration = -1.0;
        n_increasing = 0.0;

        gradient_cmsa = new GradientType(status.arcs.size(), 0.0);
        gradient_incoming = new GradientType(status.arcs.size(), 0.0);
        gradient_outgoing = new GradientType(status.arcs.size(), 0.0);

        if (options.camerini)
        {
            previous_gradient_cmsa = new GradientType(status.arcs.size(), 0.0);
            previous_gradient_incoming = new GradientType(status.arcs.size(), 0.0);
            previous_gradient_outgoing = new GradientType(status.arcs.size(), 0.0);
        }
    }

    ~Subgradient()
    {
        delete gradient_cmsa;
        delete gradient_incoming;
        delete gradient_outgoing;

        if (options.camerini)
        {
            delete previous_gradient_cmsa;
            delete previous_gradient_incoming;
            delete previous_gradient_outgoing;
        }
    }
    
    void new_iteration()
    {
        ++ iteration;
        if (!(NEARLY_EQ_TOL(iteration, 0.0)))
        {
            if (options.camerini)
            {
                previous_gradient_norm = gradient_norm;
                swap(&gradient_cmsa, &previous_gradient_cmsa);
                swap(&gradient_incoming, &previous_gradient_incoming);
                swap(&gradient_outgoing, &previous_gradient_outgoing);
            }

            std::fill(std::begin(*gradient_cmsa), std::end(*gradient_cmsa), 0.0);
            std::fill(std::begin(*gradient_incoming), std::end(*gradient_incoming), 0.0);
            std::fill(std::begin(*gradient_outgoing), std::end(*gradient_outgoing), 0.0);
        }
    }

    bool is_null()
    {
        for (unsigned i = 0u ; i < gradient_cmsa->size() ; ++i)
        {
            if (!(
                NEARLY_EQ_TOL((*gradient_cmsa)[i], (*gradient_incoming)[i]) 
                && 
                NEARLY_EQ_TOL((*gradient_cmsa)[i], (*gradient_outgoing)[i])
            ))
                return false;
        }
        return true;
    }

    void dual_has_increased()
    {
        ++ n_increasing;
    }

    void _camerini_update()
    {
        if (NEARLY_EQ_TOL(iteration, 0.0))
            return;

        double beta = 
            - options.gamma 
            * 
            (
                dot(*gradient_cmsa, *previous_gradient_cmsa)
                +
                dot(*gradient_incoming, *previous_gradient_incoming)
                +
                dot(*gradient_outgoing, *previous_gradient_outgoing)
            )
            / previous_gradient_norm
        ;

        beta = std::max(0.0, beta);

        if (NEARLY_EQ_TOL(beta, 0.0))
            return;

        for (unsigned i = 0u ; i < gradient_cmsa->size() ; ++i)
        {
            (*gradient_cmsa)[i] += beta * (*previous_gradient_cmsa)[i];
            (*gradient_incoming)[i] += beta * (*previous_gradient_incoming)[i];
            (*gradient_outgoing)[i] += beta * (*previous_gradient_outgoing)[i];
        }
    }

    unsigned update()
    {
        unsigned nb_wrong = 0u;

        if (options.camerini)
            _camerini_update();

        if (options.polyak || options.camerini)
        {
            gradient_norm = 0.0;
            for (unsigned i = 0u ; i < status.arcs.size() ; ++i)
            {
                gradient_norm += pow((*gradient_cmsa)[i], 2);
                gradient_norm += pow((*gradient_incoming)[i], 2);
                gradient_norm += pow((*gradient_outgoing)[i], 2);
            }
        }

        double stepsize = options.stepsize_scale;

        if (options.polyak)
            stepsize *= (options.polyak_wub * status.dual_weight - status.primal_weight) / gradient_norm;

        if (options.decreasing)
        {
            if (options.constant_decreasing)
                stepsize /= (1.0 + iteration);
            else
                stepsize /= (1.0 + n_increasing);
        }


        for (unsigned i = 0u ; i < status.arcs.size() ; ++i)
        {
            double mean = ((*gradient_cmsa)[i] + (*gradient_incoming)[i] + (*gradient_outgoing)[i]) / 3.0;

            if (!NEARLY_BINARY(mean))
            {
                ++ nb_wrong;
                status.cmsa_weights[i] -= stepsize * ((*gradient_cmsa)[i] - mean);
                status.incoming_weights[i] -= stepsize * ((*gradient_incoming)[i] - mean);
                status.outgoing_weights[i] -= stepsize * ((*gradient_outgoing)[i] - mean);
            }
        }

        return nb_wrong;
    }
};

