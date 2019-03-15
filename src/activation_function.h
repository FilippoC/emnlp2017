#pragma once

#include <boost/program_options.hpp>

enum struct ActivationFunction { relu, tanh, sigmoid };
struct ActivationFunctionOption{
    ActivationFunction value;

    ActivationFunctionOption(std::string const& val)
    {
        if (val == "relu")
          value = ActivationFunction::relu;
        else if (val == "tanh")
          value = ActivationFunction::tanh;
        else if (val == "sigmoid")
          value = ActivationFunction::sigmoid;
    }

    ActivationFunctionOption(): value(ActivationFunction::relu)
    {}
};

void validate(boost::any& v, 
              std::vector<std::string> const& values,
              ActivationFunctionOption* /* target_type */,
              int)
{
  using namespace boost::program_options;

  validators::check_first_occurrence(v);

  std::string const& s = validators::get_single_string(values);

  if (s == "relu" || s == "tanh" || s == "sigmoid") {
    v = boost::any(ActivationFunctionOption(s));
  } else {
    throw validation_error(validation_error::invalid_option_value);
  }
}
