//
// Created by pkwv on 5/24/18.
//

#ifndef XLEARN_PARAMETER_INITIALIZER_H
#define XLEARN_PARAMETER_INITIALIZER_H

#include <src/base/common.h>
#include <src/base/math.h>
#include <random>

namespace xLearn {

extern const int kAlign;

class ParameterInitializer {
 public:
  static ParameterInitializer* Get() {
    static ParameterInitializer initializer;
    return &initializer;
  }

  ParameterInitializer()
    : dis(0.0, 1.0) {
  }

  void InitializeW(real_t *param_w, const index_t &aux_size) {
    this->InitializeWeightOrBias(param_w, aux_size);
  }

  void InitializeBias(real_t *param_b, const index_t &aux_size) {
    this->InitializeWeightOrBias(param_b, aux_size);
  }

  /*
   * Aligned Initializer.
   * */
  void InitializeLatentFactor(real_t *param_v, const index_t &aux_size,
                              const std::string &score_func, const index_t &num_K,
                              const index_t &num_field, const real_t &scale) {
    if (score_func.compare("fm") == 0) {
      /*
       * | v_1, v_2, ... , v_n | g_1, g_2, ... , g_n |
       * */
      index_t k_aligned = ceil((real_t) num_K / kAlign) * kAlign;
      real_t coef = 1.0f / sqrt(num_K) * scale;
      real_t *w = param_v;
      for (index_t d = 0; d < num_K; ++d, ++w) {
        *w = coef * dis(generator);
      }
      for (index_t d = num_K; d < k_aligned; ++ d, ++ w) {
        *w = 0;
      }
      for (index_t d = k_aligned; d < aux_size * k_aligned; ++ d, ++ w) {
        *w = 1.0;
      }
    } else if (score_func.compare("ffm") == 0) {
      /*
       * | v_1, v_2, v_3, v_4 | g_1, g_2, g_3, g_4 | v_5, v_6, v_7, v_8 | g_5, g_6, g_7, g_8 | ...
       * */
      index_t k_aligned = ceil((real_t) num_K / kAlign) * kAlign;
      real_t* w = param_v;
      real_t coef = 1.0f / sqrt(num_K) * scale;
      for (index_t f = 0; f < num_field; ++ f) {
        for (index_t d = 0; d < k_aligned; ) {
          for (index_t s = 0; s < kAlign; ++ s, ++ w, ++ d) {
            w[0] = (d < num_K) ? coef * dis(generator) : 0.0;
            for (index_t j = 1; j < aux_size; ++ j) {
              w[kAlign * j] = 1.0;
            }
          }
          w += (aux_size - 1) * kAlign;
        }
      }
    }
  }

 private:
  std::default_random_engine generator;
  std::uniform_real_distribution<real_t> dis;

  void InitializeWeightOrBias(real_t *param, const index_t &aux_size) {
    param[0] = 0;
    for (index_t j = 1; j < aux_size; ++j) {
      param[j] = 1.0;
    }
  }

  DISALLOW_COPY_AND_ASSIGN(ParameterInitializer);
};

}
#endif //XLEARN_PARAMETER_INITIALIZER_H
