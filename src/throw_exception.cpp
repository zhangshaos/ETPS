#include "throw_exception.h"
#include <cpptrace/cpptrace.hpp> //str_trace()
#include <stdexcept>


void
throw_runtime_error(const std::string &msg, uint32_t skip){
  auto trace_msg = str_trace(msg, skip + 1);
  throw std::runtime_error(trace_msg);
}
