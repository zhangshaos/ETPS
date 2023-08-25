#ifndef CHECK_FLOAT_ERROR_H_
#define CHECK_FLOAT_ERROR_H_

#ifndef N_SELF_CHECK_FLOAT
#include <exception>
#include <cstdint>
#include "throw_exception.h"

template<typename T>
struct Float {
  T f_number;

  static inline
  void
  check_and_error(T f){
    if (std::isnan(f))
      throw_runtime_error("float-point number found nan!", 1);
    else if (std::isinf(f))
      throw_runtime_error("float-point number found inf!", 1);
  }

  Float() : f_number(std::numeric_limits<T>::signaling_NaN()){

  }

  Float(T f)
      : f_number(f) {
    check_and_error(f_number);
  }
  Float(const Float &other)
      : f_number(other.f_number) {
    check_and_error(f_number);
  }
  Float(Float &&other)
      : f_number(other.f_number) {
    check_and_error(f_number);
  }
  Float& operator=(const Float& other) {
    f_number = other.f_number;
    return *this;
  }
  Float& operator=(Float&& other) {
    f_number = other.f_number;
    return *this;
  }

  operator T() const {
    return f_number;
  }
  operator Float<float>()const {
    return f_number;
  }
  operator Float<double>()const {
    return f_number;
  }

  Float operator*(Float other) const {
    T result = f_number * other.f_number;
    return Float(result);
  }
  Float& operator*=(Float other) {
    operator=(operator*(other));
    return *this;
  }

  Float operator/(Float other) const {
    if (other.f_number == 0.0)
      throw_runtime_error("float-point number divided by 0!");
    T result = f_number / other.f_number;
    return Float(result);
  }
  Float& operator/=(Float other) {
    operator=(operator/(other));
    return *this;
  }

  Float operator+(Float other)const{
    T result = f_number + other.f_number;
    return Float(result);
  }
  Float& operator+=(Float other){
    operator=(operator+(other));
    return *this;
  }

  Float operator-(Float other)const{
    T result = f_number - other.f_number;
    return Float(result);
  }
  Float& operator-=(Float other){
    operator=(operator-(other));
    return *this;
  }
};

using F32 = Float<float>;
using F64 = Float<double>;
#else
using F32 = float;
using F64 = double;
#endif

#endif //CHECK_FLOAT_ERROR_H_
