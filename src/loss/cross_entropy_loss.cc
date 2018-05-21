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
This file is the implementation of CrossEntropyLoss class.
*/

#include "src/loss/cross_entropy_loss.h"
#include "ps-lite/include/ps/kv_app.h"

#include <atomic>
#include <thread>

namespace xLearn {

// Calculate loss in one thread.
static void ce_evalute_thread(const std::vector<real_t>* pred,
                              const std::vector<real_t>* label,
                              real_t* tmp_sum,
                              size_t start_idx,
                              size_t end_idx) {
  CHECK_GE(end_idx, start_idx);
  *tmp_sum = 0;
  for (size_t i = start_idx; i < end_idx; ++i) {
    real_t y = (*label)[i] > 0 ? 1.0 : -1.0;
    (*tmp_sum) += log1p(exp(-y*(*pred)[i]));
  }
}

//------------------------------------------------------------------------------
// Calculate loss in multi-thread:
//
//                         master_thread
//                      /       |         \
//                     /        |          \
//                thread_1    thread_2    thread_3
//                   |           |           |
//                    \          |           /
//                     \         |          /
//                       \       |        /
//                         master_thread
//------------------------------------------------------------------------------
void CrossEntropyLoss::Evalute(const std::vector<real_t>& pred,
                               const std::vector<real_t>& label) {
  CHECK_NE(pred.empty(), true);
  CHECK_NE(label.empty(), true);
  total_example_ += pred.size();
  // multi-thread training
  std::vector<real_t> sum(threadNumber_, 0);
  for (int i = 0; i < threadNumber_; ++i) {
    size_t start_idx = getStart(pred.size(), threadNumber_, i);
    size_t end_idx = getEnd(pred.size(), threadNumber_, i);
    pool_->enqueue(std::bind(ce_evalute_thread,
                             &pred,
                             &label,
                             &(sum[i]),
                             start_idx,
                             end_idx));
  }
  // Wait all of the threads finish their job
  pool_->Sync(threadNumber_);
  // Accumulate loss
  for (size_t i = 0; i < sum.size(); ++i) {
    loss_sum_ += sum[i];
  }
}


// Calculate gradient in one thread.
static void ce_gradient_thread(const DMatrix* matrix,
                               Model* model,
                               Score* score_func,
                               bool is_norm,
                               real_t* sum,
                               size_t start_idx,
                               size_t end_idx) {
  CHECK_GE(end_idx, start_idx);
  *sum = 0;
  for (size_t i = start_idx; i < end_idx; ++i) {
    SparseRow* row = matrix->row[i];
    real_t norm = is_norm ? matrix->norm[i] : 1.0;
    real_t pred = score_func->CalcScore(row, *model, norm);
    // partial gradient
    real_t y = matrix->Y[i] > 0 ? 1.0 : -1.0;
    *sum += log1p(exp(-y*pred));
    real_t pg = -y/(1.0+(1.0/exp(-y*pred)));
    // real gradient and update
    score_func->CalcGrad(row, *model, pg, norm);
  }
}

// Calculate gradient in one thread.
static void ce_gradient_thread_mini_batch(DMatrix* matrix,
                               Model* model,
                               Score* score_func,
                               bool is_norm,
                               real_t* sum,
                               size_t start_idx,
                               size_t end_idx,
                               index_t batch_size,
                               int thread_id,
                               index_t bias_idx) {
  CHECK_GE(end_idx, start_idx);
  *sum = 0;
  ps::KVWorker<real_t> kv_w(0, thread_id);
  ps::KVWorker<real_t> kv_v(1, thread_id);
  //index_t bias_idx = model->GetNumFeature() - 1;
  for (size_t i = start_idx; i < end_idx; i += batch_size) {
    size_t i_end_idx = std::min(end_idx, i + batch_size);
    index_t sample_num = i_end_idx - i;
    std::vector<ps::Key> dense_to_sparse;
    // to dense
    matrix->ToDenseMatrix<ps::Key>(i, i_end_idx, dense_to_sparse);
    index_t feat_num = dense_to_sparse.size();
    std::vector<ps::Key> feat_idx = dense_to_sparse;
    // add bias
    feat_idx.push_back(bias_idx);
    //std::vector<real_t> param_w(feat_idx.size());
    Vector<real_t> param_w(feat_idx.size(), model->GetNumParameter_w(), model->GetParameter_w());
    auto kv_w_ts = kv_w.Pull(feat_idx, &param_w);
    kv_w.Wait(kv_w_ts);
    //std::vector<real_t> param_v;
    Vector<real_t> param_v(dense_to_sparse.size(), model->GetNumParameter_v(), model->GetParameter_v());
    if (model->GetScoreFunction().compare("fm") == 0
        || model->GetScoreFunction().compare("ffm") == 0) {
      param_v.resize(dense_to_sparse.size()
                     * model->GetNumK()
                     * model->GetNumField());
      auto kv_v_ts = kv_v.Pull(dense_to_sparse, &param_v);
      kv_v.Wait(kv_v_ts);
    }
    //kv_w.Wait(kv_w_ts);
    //model->SetParamW(param_w.data(), param_w.size() - 1);
    //model->SetParamB(param_w.data() + feat_num);
    //model->SetParamV(param_v.data());
    model->SetParamB(param_w.data() + feat_num);

    real_t pg_sum = 0.0;
    for (int j = i; j < i_end_idx; ++ j) {
      SparseRow* row = matrix->row[j];
      real_t norm = is_norm ? matrix->norm[i] : 1.0;
      real_t pred = score_func->CalcScore(row, *model, norm);
      // partial gradient
      real_t y = matrix->Y[i] > 0 ? 1.0 : -1.0;
      *sum += log1p(exp(-y*pred));
      pg_sum += -y/(1.0+(1.0/exp(-y*pred)));
    }
    real_t pg = pg_sum / sample_num;
    std::vector<real_t> gradient_w(param_w.size(), 0.0);
    std::vector<real_t> gradient_v(param_v.size(), 0.0);
    for (int j = i; j < i_end_idx; ++ j) {
      SparseRow* row = matrix->row[j];
      real_t norm = is_norm ? matrix->norm[i] : 1.0;
      score_func->CalcGrad(row, *model, pg, gradient_w, gradient_v, norm);
    }
    // to sparse
    matrix->ToSparseMatrix(i, i_end_idx, dense_to_sparse);
    kv_w_ts = kv_w.Push(feat_idx, gradient_w);
    if (model->GetScoreFunction().compare("fm") == 0
        || model->GetScoreFunction().compare("ffm") == 0) {
      kv_v.Wait(kv_v.Push(dense_to_sparse, gradient_v));
    }
    kv_w.Wait(kv_w_ts);
  }
}

//------------------------------------------------------------------------------
// Calculate gradient in multi-thread
//
//                         master_thread
//                      /       |         \
//                     /        |          \
//                thread_1    thread_2    thread_3
//                   |           |           |
//                    \          |           /
//                     \         |          /
//                       \       |        /
//                         master_thread
//------------------------------------------------------------------------------
void CrossEntropyLoss::CalcGrad(const DMatrix* matrix,
                                Model& model) {
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_length, 0);
  size_t row_len = matrix->row_length;
  total_example_ += row_len;
  // multi-thread training
  int count = lock_free_ ? threadNumber_ : 1;
  std::vector<real_t> sum(count, 0);
  for (int i = 0; i < count; ++i) {
    index_t start_idx = getStart(row_len, count, i);
    index_t end_idx = getEnd(row_len, count, i);
    pool_->enqueue(std::bind(ce_gradient_thread,
                             matrix,
                             &model,
                             score_func_,
                             norm_,
                             &(sum[i]),
                             start_idx,
                             end_idx));
  }
  // Wait all of the threads finish their job
  pool_->Sync(count);
  // Accumulate loss
  for (int i = 0; i < sum.size(); ++i) {
    loss_sum_ += sum[i];
  }
}

// Given data sample and current model, calculate gradient.
// Note that this method doesn't update local model, and the
// gradient will be pushed to the parameter server, which is 
// used for distributed computation.
void CrossEntropyLoss::CalcGradDist(DMatrix* matrix,
                                    Model& model,
                                    std::vector<real_t>& grad) {
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_length, 0);
  size_t row_len = matrix->row_length;
  total_example_ += row_len;
  // multi-thread training
  int count = lock_free_ ? threadNumber_ : 1;
  std::vector<real_t> sum(count, 0);
  std::vector<Model> model_arr(count);
  index_t batch_feat_num = matrix->UniqueFeatureNum(0, batch_size_);
  for (int i = 0; i < count; ++i) {
    index_t start_idx = getStart(row_len, count, i);
    index_t end_idx = getEnd(row_len, count, i);
    Model& model_i = model_arr[i];
    model_i.Initialize(model.GetScoreFunction(), model.GetLossFunction(),
                       batch_feat_num * 1.2, model.GetNumField(),
                       model.GetNumK(), model.GetAuxiliarySize(),
                       model.GetScale());
    pool_->enqueue(std::bind(ce_gradient_thread_mini_batch,
                             matrix,
                             &model_i,
                             score_func_,
                             norm_,
                             &(sum[i]),
                             start_idx,
                             end_idx,
                             batch_size_,
                             i,
                             model.GetNumParameter_w() - 1));
  }
  // Wait all of the threads finish their job
  pool_->Sync(count);
  // Accumulate loss
  for (int i = 0; i < sum.size(); ++i) {
    loss_sum_ += sum[i];
  }
}

} // namespace xLearn
