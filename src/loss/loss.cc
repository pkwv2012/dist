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
This file is the implementation of the base Loss class.
*/

#include <src/distributed/parameter_server.h>
#include "ps-lite/include/ps/kv_app.h"
#include "src/loss/loss.h"
#include "src/loss/squared_loss.h"
#include "src/loss/cross_entropy_loss.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(xLearn_loss_registry, Loss);
REGISTER_LOSS("squared", SquaredLoss);
REGISTER_LOSS("cross-entropy", CrossEntropyLoss);

// Predict in one thread
void pred_thread(const DMatrix* matrix,
                 Model* model,
                 std::vector<real_t>* pred,
                 Score* score_func_,
                 bool is_norm,
                 size_t start_idx,
                 size_t end_idx) {
  CHECK_GE(end_idx, start_idx);
  for (size_t i = start_idx; i < end_idx; ++i) {
    SparseRow* row = matrix->row[i];
    real_t norm = is_norm ? matrix->norm[i] : 1.0;
    (*pred)[i] = score_func_->CalcScore(row, *model, norm);
  }
}

// Predict in one thread
void pred_thread_mini_batch(DMatrix* matrix,
                            Model* model,
                            std::vector<real_t>* pred,
                            Score* score_func_,
                            bool is_norm,
                            size_t start_idx,
                            size_t end_idx,
                            size_t batch_size,
                            int thread_id,
                            index_t bias_idx) {
  CHECK_GE(end_idx, start_idx);
  KVStore xlearn_worker(thread_id);
  for (size_t i = start_idx; i < end_idx; i += batch_size) {
    size_t i_end_idx = std::min(end_idx, i + batch_size);
    index_t sample_num = i_end_idx - i;
    std::vector<ps::Key> dense_to_sparse;
    // to dense
    matrix->ToDenseMatrix<ps::Key>(i, i_end_idx, dense_to_sparse);
    Model model_i;
    model_i.Initialize(model->GetScoreFunction(), model->GetLossFunction(),
                       dense_to_sparse.size(), model->GetNumField(), model->GetNumK(),
                       model->GetAuxiliarySize());
    // add bias
    dense_to_sparse.push_back(bias_idx);
    xlearn_worker.Pull(dense_to_sparse, &model_i);

    for (int j = i; j < i_end_idx; ++ j) {
      SparseRow* row = matrix->row[j];
      real_t norm = is_norm ? matrix->norm[i] : 1.0;
      (*pred)[j] = score_func_->CalcScore(row, model_i, norm);
    }
    // to sparse
    matrix->ToSparseMatrix(i, i_end_idx, dense_to_sparse);
  }
}

// Predict in multi-thread
void Loss::Predict(const DMatrix* matrix,
                   Model& model,
                   std::vector<real_t>& pred) {
  CHECK_NOTNULL(matrix);
  CHECK_NE(pred.empty(), true);
  CHECK_EQ(pred.size(), matrix->row_length);
  index_t row_len = matrix->row_length;
  // Predict in multi-thread
  for (int i = 0; i < threadNumber_; ++i) {
    size_t start_idx = getStart(row_len, threadNumber_, i);
    size_t end_idx = getEnd(row_len, threadNumber_, i);
    pool_->enqueue(std::bind(pred_thread,
                             matrix,
                             &model,
                             &pred,
                             score_func_,
                             norm_,
                             start_idx,
                             end_idx));
  }
  // Wait all of the threads finish their job
  pool_->Sync(threadNumber_);
}

void Loss::PredictDist(DMatrix* matrix,
                       Model& model,
                       std::vector<real_t>& pred) {
  CHECK_NOTNULL(matrix);
  CHECK_NE(pred.empty(), true);
  CHECK_EQ(pred.size(), matrix->row_length);
  index_t row_len = matrix->row_length;
  index_t batch_feature_num = matrix->UniqueFeatureNum(0, batch_size_);
  std::vector<Model> model_arr(threadNumber_);
  // Predict in multi-thread
  for (int i = 0; i < threadNumber_; ++i) {
    size_t start_idx = getStart(row_len, threadNumber_, i);
    size_t end_idx = getEnd(row_len, threadNumber_, i);
    Model &model_i = model_arr[i];
    model_i.Initialize(model.GetScoreFunction(), model.GetLossFunction(),
                       batch_feature_num * 1.2, model.GetNumField(),
                       model.GetNumK(), model.GetAuxiliarySize(),
                       model.GetScale());
    pool_->enqueue(std::bind(pred_thread_mini_batch,
                             matrix,
                             &model,
                             &pred,
                             score_func_,
                             norm_,
                             start_idx,
                             end_idx,
                             batch_size_,
                             i,
                             model.GetNumFeature() - 1));
  }
  // Wait all of the threads finish their job
  pool_->Sync(threadNumber_);
}

// Calculate gradient in one thread.
static void gradient_thread_mini_batch(DMatrix* matrix,
                                          Model* model,
                                          Model* gradient,
                                          Score* score_func,
                                          bool is_norm,
                                          real_t* sum,
                                          size_t start_idx,
                                          size_t end_idx,
                                          index_t batch_size,
                                          int thread_id,
                                          index_t bias_idx,
                                          std::function<real_t(const real_t&, const real_t&)> calc_loss,
                                          std::function<real_t(const real_t&, const real_t&)> calc_pg) {
  CHECK_GE(end_idx, start_idx);
  *sum = 0;
  KVStore xlearn_worker(thread_id);
  //index_t bias_idx = model->GetNumFeature() - 1;
  for (size_t i = start_idx; i < end_idx; i += batch_size) {
    size_t i_end_idx = std::min(end_idx, i + batch_size);
    index_t sample_num = i_end_idx - i;
    std::vector<ps::Key> dense_to_sparse;
    // to dense
    matrix->ToDenseMatrix<ps::Key>(i, i_end_idx, dense_to_sparse);
    Model model_i;
    model_i.Initialize(model->GetScoreFunction(), model->GetLossFunction(),
                       dense_to_sparse.size(), model->GetNumField(), model->GetNumK(),
                       model->GetAuxiliarySize());
    Model gradient_i;
    gradient_i.Initialize(model->GetScoreFunction(), model->GetLossFunction(),
                       dense_to_sparse.size(), model->GetNumField(), model->GetNumK(),
                       model->GetAuxiliarySize());
    gradient_i.SetZero();
    // add bias
    dense_to_sparse.push_back(bias_idx);
    xlearn_worker.Pull(dense_to_sparse, &model_i);

    real_t pg_sum = 0.0;
    for (int j = i; j < i_end_idx; ++ j) {
      SparseRow* row = matrix->row[j];
      real_t norm = is_norm ? matrix->norm[i] : 1.0;
      real_t pred = score_func->CalcScore(row, model_i, norm);
      // partial gradient
      real_t y = matrix->Y[i] > 0 ? 1.0 : -1.0;
      //*sum += log1p(exp(-y*pred));
      *sum += calc_loss(matrix->Y[i], pred);
      //pg_sum += -y/(1.0+(1.0/exp(-y*pred)));
      pg_sum += calc_pg(matrix->Y[i], pred);
    }
    real_t pg = pg_sum / sample_num;
    for (int j = i; j < i_end_idx; ++ j) {
      SparseRow* row = matrix->row[j];
      real_t norm = is_norm ? matrix->norm[i] : 1.0;
      score_func->CalcGrad(row, model_i, gradient_i, pg, norm);
    }
    // to sparse
    matrix->ToSparseMatrix(i, i_end_idx, dense_to_sparse);
    xlearn_worker.Push(dense_to_sparse, gradient_i);
  }
}

// Given data sample and current model, calculate gradient.
// Note that this method doesn't update local model, and the
// gradient will be pushed to the parameter server, which is 
// used for distributed computation.
void Loss::CalcGradDist(DMatrix* matrix,
                        Model& model,
                        std::vector<real_t>& grad) {
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_length, 0);
  size_t row_len = matrix->row_length;
  total_example_ += row_len;
  // multi-thread training
  int count = lock_free_ ? threadNumber_ : 1;
  std::vector<real_t> sum(count, 0);
  std::vector<Model> model_arr(count), gradient_arr(count);
  index_t batch_feat_num = matrix->UniqueFeatureNum(0, batch_size_);
  for (int i = 0; i < count; ++i) {
    index_t start_idx = getStart(row_len, count, i);
    index_t end_idx = getEnd(row_len, count, i);
    // model_i is used for store the partial model of the mini-batch.
    Model& model_i = model_arr[i];
    model_i.Initialize(model.GetScoreFunction(), model.GetLossFunction(),
                       batch_feat_num * 1.2, model.GetNumField(),
                       model.GetNumK(), model.GetAuxiliarySize(),
                       model.GetScale());
    // gradient_i is used for store gradient.
    Model& gradient_i = gradient_arr[i];
    gradient_i.Initialize(model.GetScoreFunction(), model.GetLossFunction(),
                          batch_feat_num * 1.2, model.GetNumField(),
                          model.GetNumK(), model.GetAuxiliarySize(),
                          model.GetScale());
    pool_->enqueue(std::bind(gradient_thread_mini_batch,
                             matrix,
                             &model,
                             &gradient_i,
                             score_func_,
                             norm_,
                             &(sum[i]),
                             start_idx,
                             end_idx,
                             batch_size_,
                             i,
                             model.GetNumParameter_w() - 1,
                             [this] (const real_t& lhs, const real_t& rhs) {
                               return this->CalcLoss(lhs, rhs);
                             },
                             [this] (const real_t& lhs, const real_t& rhs) {
                               return this->CalcPartialGradient(lhs, rhs);
                             }));

  }
  // Wait all of the threads finish their job
  pool_->Sync(count);
  // Accumulate loss
  for (int i = 0; i < sum.size(); ++i) {
    loss_sum_ += sum[i];
  }
}

}  // namespace xLearn
