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
This file defines the CrossEntropyLoss class.
*/

#ifndef XLEARN_LOSS_DIST_CROSS_ENTROPY_LOSS_H_
#define XLEARN_LOSS_DIST_CROSS_ENTROPY_LOSS_H_

#include "src/base/common.h"
#include "src/distributed/dist_loss.h"

namespace xLearn {

//------------------------------------------------------------------------------
// CrossEntropyLoss is used for classification tasks, which
// has the following form:
// loss = sum_all_example(log(1.0+exp(-y*pred)))
//------------------------------------------------------------------------------
class DistCrossEntropyLoss : public DistLoss {
 public:
  // Constructor and Desstructor
  DistCrossEntropyLoss() { }
  ~DistCrossEntropyLoss() { }

  void Predict(const DMatrix* data_matrix,
      Model& model,
      std::vector<real_t>& pred);

  // Given predictions and labels, accumulate cross-entropy loss.
  void Evalute(const std::vector<real_t>& pred,
               const std::vector<real_t>& label);

  // Given data sample and current model, calculate gradient
  // and update current model parameters.
  // This function will also accumulate the loss value.
  void CalcGrad(DMatrix* data_matrix, Model& model);
  //void DistCalcGrad(const DMatrix& data_matrix, Model& model);

  // Return current loss type.
  std::string loss_type() { return "dist_log_loss"; }

 private:
  DISALLOW_COPY_AND_ASSIGN(DistCrossEntropyLoss);
};

}  // namespace xLearn

#endif  // XLEARN_LOSS_DIST_CROSS_ENTROPY_LOSS_H_
