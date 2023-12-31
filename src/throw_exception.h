#ifndef ETPS_SRC_THROW_EXCEPTION_H_
#define ETPS_SRC_THROW_EXCEPTION_H_

#include <cstdint>
#include <string>

void
throw_runtime_error(const std::string &msg, uint32_t skip=0);

#endif //ETPS_SRC_THROW_EXCEPTION_H_
