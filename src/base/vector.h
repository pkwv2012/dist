//
// Created by pkwv on 5/22/18.
//

#ifndef XLEARN_VECTOR_H
#define XLEARN_VECTOR_H


#include <cstddef>

namespace xLearn {

template<class Val>
class Vector {
public:
  Vector(size_t size, size_t capacity, Val *ptr)
    : size_(size), capacity_(capacity), ptr_(ptr) {
  }

  bool empty() {
    return size_ == 0;
  }

  size_t size() {
    return size_;
  }

  Val *data() {
    return ptr_;
  }

  void resize(size_t size) {
    if (size <= capacity_) {
      size_ = size;
    } else {
    }
  }

private:
  size_t size_;
  size_t capacity_;
  Val *ptr_;
};

}

#endif //XLEARN_VECTOR_H
