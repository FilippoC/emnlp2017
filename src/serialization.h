#pragma once

#include <fstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

template<typename Type>
void save_object(const std::string path, const Type& obj)
{
    std::ofstream out(path);
    boost::archive::text_oarchive oa(out);
    oa << obj;
    out.close();
}

template<typename Type>
void read_object(const std::string path, Type& obj)
{
    std::ifstream in(path);
    boost::archive::text_iarchive ia(in);
    ia >> obj;
    in.close();
}


