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
This file defines the FMrScore (factorization machine) class.
*/

#ifndef XLEARN_DIST_FM_SCORE_H_
#define XLEARN_DIST_FM_SCORE_H_

#include "src/base/common.h"
#include "src/data/model_parameters.h"
#include "src/distributed/dist_score_function.h"

namespace xLearn {

//------------------------------------------------------------------------------
// FMScore is used to implemente factorization machines, in which
// the socre function is y = sum( (V_i*V_j)(x_i * x_j) )
// Here we leave out the linear term and bias term.
//------------------------------------------------------------------------------
class DistFMScore : public DistScore {
 public:
  // Constructor and Desstructor
  DistFMScore() { }
  ~DistFMScore() { }

  // Given one exmaple and current model, this method
  // returns the fm score.
  real_t CalcScore(const SparseRow* row,
                   Model& model,
                   std::map<index_t, real_t>* w,
                   std::map<index_t, std::vector<real_t>>* v,
                   real_t norm = 1.0);

  // Calculate gradient and update current
  // model parameters.
  void DistCalcGrad(const DMatrix* matrix,
                    Model& model,
                    std::map<index_t, real_t>& w,
                    std::map<index_t, std::vector<real_t>>& v,
                    real_t* sum,
                    std::map<index_t, real_t>& w_g,
                    std::map<index_t, std::vector<real_t>>& v_g,
                    index_t start_idx,
                    index_t end_idx
                   );

 protected:
  // Calculate gradient and update model using sgd
  void calc_grad_sgd(const DMatrix* matrix,
                     Model& model,
                     std::map<index_t, real_t>& w,
                     std::map<index_t, std::vector<real_t>>& v,
                     real_t* sum,
                     std::map<index_t, real_t>& w_g,
                     std::map<index_t, std::vector<real_t>>& v_g,
                     index_t start_idx,
                     index_t end_idx
                    );

  // Calculate gradient and update model using adagrad
  void calc_grad_adagrad(const DMatrix* matrix,
                         Model& model,
                         std::map<index_t, real_t>& w,
                         std::map<index_t, std::vector<real_t>>& v,
                         real_t* sum,
                         std::map<index_t, real_t>& w_g,
                         std::map<index_t, std::vector<real_t>>& v_g,
                         index_t start_idx,
                         index_t end_idx
                        );

  // Calculate gradient and update model using ftrl
  void calc_grad_ftrl(const DMatrix* matrix,
                      Model& model,
                      std::map<index_t, real_t>& w,
                      std::map<index_t, std::vector<real_t>>& v,
                      real_t* sum,
                      std::map<index_t, real_t>& w_g,
                      std::map<index_t, std::vector<real_t>>& v_g,
                      index_t start_idx,
                      index_t end_idx
                      );
 private:
  real_t* comp_res = nullptr;
  real_t* comp_z_lt_zero = nullptr;
  real_t* comp_z_gt_zero = nullptr;

 private:
  DISALLOW_COPY_AND_ASSIGN(DistFMScore);
};

} // namespace xLearn

#endif // XLEARN_DIST_FM_SCORE_H_
