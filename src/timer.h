#pragma once

#include <chrono>

struct Timer
{
    bool _running = false;
    double _total = 0.0;
    decltype(std::chrono::steady_clock::now()) _begin;

    void start()
    {
        assert(!_running);
        _running = true;
        _begin = std::chrono::steady_clock::now();
    }

    void stop(bool check_running=true)
    {
        if (check_running)
            assert(_running);

        _running = false;

        auto end = std::chrono::steady_clock::now();
        _total +=
            std::chrono::duration_cast<std::chrono::milliseconds>(
                end - _begin
            ).count()
        ;
    }

    double milliseconds() const
    {
        assert(!_running);
        return _total;
    }

    double seconds() const
    {
        assert(!_running);
        return _total / 1000.0;
    }
};

