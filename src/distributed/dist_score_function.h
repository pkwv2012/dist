//------------------------------------------------------------------------------
// Copyright (c) 2016 by contributors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//------------------------------------------------------------------------------

/*
Author: Chao Ma (mctt90@gmail.com)
This file defines the DistScore class, including linear score,
FM score, FFM score, and etc.
*/

#ifndef XLEARN_LOSS_DIST_SCORE_FUNCTION_H_
#define XLEARN_LOSS_DIST_SCORE_FUNCTION_H_

#include <vector>
#include <unordered_map>
#include <map>

#include "src/base/common.h"
#include "src/base/class_register.h"
#include "src/data/data_structure.h"
#include "src/data/hyper_parameters.h"
#include "src/data/model_parameters.h"

namespace xLearn {

//------------------------------------------------------------------------------
// DistScore is an abstract class, which can be implemented by different
// score functions such as LinearScore (liner_score.h), FMScore (fm_score.h)
// FFMScore (ffm_score.h) etc. On common, we initial a Score function and
// pass its pointer to a Loss class like this:
//
//  Score* score = new FMScore();
//  score->CalcScore(row, model, norm);
//  score->CalcGrad(row, model, pg, norm);
//
// In general, the CalcGrad() will be used in loss function.
//------------------------------------------------------------------------------
class DistScore {
 public:
  // Constructor and Desstructor
  DistScore() { }
  virtual ~DistScore() { }

  // Invoke this function before we use this class.
  virtual void Initialize(real_t learning_rate,
                          real_t regu_lambda,
                          real_t alpha,
                          real_t beta,
                          real_t lambda_1,
                          real_t lambda_2,
                          std::string& opt_type) {
    learning_rate_ = learning_rate;
    regu_lambda_ = regu_lambda;
    alpha_ = alpha;
    beta_ = beta;
    lambda_1_ = lambda_1;
    lambda_2_ = lambda_2;
    opt_type_ = opt_type;
  }

  // Given one exmaple and current model, this method
  // returns the score
  virtual real_t CalcScore(const SparseRow* row,
                           std::map<index_t, real_t>* w,
                           std::map<index_t, std::vector<real_t>>* v,
                           real_t norm = 1.0) = 0;

  virtual void DistCalcGrad(const DMatrix* matrix,
                               std::map<index_t, real_t>& w,
                               std::map<index_t, std::vector<real_t>>* v,
                               real_t* sum,
                               std::map<index_t, real_t>& g,
                               std::map<index_t, real_t>& v_g,
                               index_t start_idx,
                               index_t end_idx) = 0;

 protected:
  real_t learning_rate_;
  real_t regu_lambda_;
  real_t alpha_;
  real_t beta_;
  real_t lambda_1_;
  real_t lambda_2_;
  std::string opt_type_;

 private:
  DISALLOW_COPY_AND_ASSIGN(DistScore);
};

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_DEFINE_REGISTRY(xLearn_dist_score_registry, DistScore);

#define REGISTER_SCORE(format_name, score_name)             \
  CLASS_REGISTER_OBJECT_CREATOR(                            \
      xLearn_dist_score_registry,                                \
      DistScore,                                                \
      format_name,                                          \
      score_name)

#define CREATE_DIST_SCORE(format_name)                           \
  CLASS_REGISTER_CREATE_OBJECT(                             \
      xLearn_dist_score_registry,                                \
      format_name)

}  // namespace xLearn

#endif  // XLEARN_LOSS_DIST_SCORE_FUNCTION_H_
