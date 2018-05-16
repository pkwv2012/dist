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

#include "src/distributed/dist_cross_entropy_loss.h"

#include<thread>
#include<atomic>
#include<map>
#include <src/base/timer.h>
#include <src/base/format_print.h>
#include <unordered_set>

namespace xLearn {

static void pred_thread(const DMatrix* data_matrix,
                        Model* model,
                        std::unordered_map<index_t, real_t>& w,
                        std::unordered_map<index_t, std::vector<real_t>>& v,
                        std::vector<real_t>* pred,
                        DistScore* dist_score_func,
                        bool is_norm,
                        size_t start_idx,
                        size_t end_idx) {
  CHECK_GE(end_idx, start_idx);
  for (size_t i = start_idx; i < end_idx; ++i) {
    SparseRow* row = data_matrix->row[i];
    real_t norm = is_norm ? data_matrix->norm[i] : 1.0;
    (*pred)[i] = dist_score_func->CalcScore(row, *model, &w, &v, norm);
  }
}

static void mini_batch_pred_thread(const DMatrix* data_matrix,
                                   Model* model,
                                   std::vector<real_t>* pred,
                                   DistScore* dist_score_func,
                                   bool is_norm,
                                   size_t start_idx,
                                   size_t end_idx,
                                   int thread_id) {
  LOG(INFO) << "mini_batch_pred_thread||thread_id=" << thread_id << std::endl;
  std::vector<ps::Key> feature_ids = data_matrix->GetSortedUniqueIndex<ps::Key>(start_idx, end_idx);
  auto gradient_pull = std::make_shared<std::vector<float>>();
  auto v_pull = std::make_shared<std::vector<float>>();
  std::unordered_map<index_t, real_t> weight_map;
  std::unordered_map<index_t, std::vector<real_t>> v_map;

  LOG(INFO) << "score func " << model->GetScoreFunction() << std::endl;
  LOG(INFO) << "prepare info" << std::endl;

  gradient_pull->resize(feature_ids.size());
  LOG(INFO) << "waiting for pull params" << std::endl;
  auto kv_w_ = std::make_shared<ps::KVWorker<float>>(0, thread_id);
  kv_w_->Wait(kv_w_->Pull(feature_ids, &(*gradient_pull)));
  int num_K = model->GetNumK();
  int num_field = model->GetNumField();
  int len = num_field * num_K;
  v_pull->resize(feature_ids.size() * len);
  auto kv_v_ = std::make_shared<ps::KVWorker<float>>(1, thread_id);
  if (model->GetScoreFunction().compare("fm") == 0 ||
      model->GetScoreFunction().compare("ffm") == 0) {
    kv_v_->Wait(kv_v_->Pull(feature_ids, &(*v_pull)));
  }
  LOG(INFO) << "got params" << std::endl;
  for (int i = 0; i < gradient_pull->size(); ++i) {
    index_t idx = feature_ids[i];
    real_t weight = (*gradient_pull)[i];
    weight_map[idx] = weight;
  }
  LOG(INFO) << "model NumK=" << model->GetNumK() << std::endl;
  LOG(INFO) << "val size=" << v_pull->size() << std::endl;
  for (int i = 0; i < feature_ids.size(); ++i) {
    index_t idx = feature_ids[i];
    std::vector<real_t> vec_k;
    for(int j = 0; j < model->GetNumK(); ++j) {
      vec_k.push_back((*v_pull)[i * model->GetNumK() + j]);
    }
    v_map[idx] = vec_k;
  }
  pred_thread(data_matrix, model, weight_map, v_map, pred, dist_score_func, is_norm, start_idx, end_idx);
}

void DistCrossEntropyLoss::Predict(const DMatrix* data_matrix,
                                   Model& model,
                                   std::vector<real_t>& pred) {
  CHECK_NOTNULL(data_matrix);
  CHECK_GT(data_matrix->row_length, 0);
  int count = std::ceil(data_matrix->row_length * 1.0 / batch_size_);
  for(int i = 0; i < count; ++ i) {
    // Get a mini-batch from current data matrix
    // Pull the model parameter from parameter server
    int ti = i;
    size_t start_idx = i * batch_size_;
    size_t end_idx = std::min((i + 1) * batch_size_, data_matrix->row_length);
    pool_->enqueue(std::bind(mini_batch_pred_thread,
                             data_matrix,
                             &model,
                             &pred,
                             dist_score_func_,
                             norm_,
                             start_idx,
                             end_idx,
                             ti));
  }
  pool_->Sync(count);
}

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
void DistCrossEntropyLoss::Evalute(const std::vector<real_t>& pred,
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
                               std::unordered_map<index_t, real_t>& w,
                               std::unordered_map<index_t, std::vector<real_t>>& v,
                               DistScore* dist_score_func,
                               bool is_norm,
                               real_t* sum,
                               std::unordered_map<index_t, real_t>& w_g,
                               std::unordered_map<index_t, std::vector<real_t>>& v_g,
                               size_t start_idx,
                               size_t end_idx) {
  CHECK_GE(end_idx, start_idx);
  CHECK_NOTNULL(dist_score_func);
  LOG(INFO) << "ce_gradient_thread" << std::endl;
  LOG(INFO) << dist_score_func->GetOptType() << std::endl;
  dist_score_func->DistCalcGrad(matrix, *model, w, v, sum, w_g, v_g, start_idx, end_idx);
}

static void mini_batch_gradient_thread(const DMatrix *matrix,
                        Model* model,
                        DistScore* dist_score_func,
                        bool is_norm,
                        real_t* sum,
                        int thread_id) {
  std::vector<ps::Key> feature_ids = matrix->GetSortedUniqueIndex<ps::Key>();
  auto gradient_pull = std::make_shared<std::vector<float>>();
  auto v_pull = std::make_shared<std::vector<float>>();
  std::unordered_map<index_t, real_t> weight_map;
  std::unordered_map<index_t, std::vector<real_t>> v_map;

  std::unordered_map<index_t, real_t> gradient_push_map;
  std::unordered_map<index_t, std::vector<real_t>> v_push_map;

  auto gradient_push = std::make_shared<std::vector<float>>();
  auto v_push = std::make_shared<std::vector<float>>();
  LOG(INFO) << "score func " << model->GetScoreFunction() << std::endl;
  LOG(INFO) << "prepare info" << std::endl;

  gradient_pull->resize(feature_ids.size());
  LOG(INFO) << "waiting for pull params" << std::endl;
  auto kv_w_ = std::make_shared<ps::KVWorker<float>>(0, thread_id);
  kv_w_->Wait(kv_w_->Pull(feature_ids, &(*gradient_pull)));
  auto kv_v_ = std::make_shared<ps::KVWorker<float>>(1, thread_id);
  if (model->GetScoreFunction().compare("fm") == 0) {
    v_pull->resize(feature_ids.size() * model->GetNumK());
    kv_v_->Wait(kv_v_->Pull(feature_ids, &(*v_pull)));
  } else if (model->GetScoreFunction().compare("ffm") == 0) {
    v_pull->resize(feature_ids.size() * model->GetNumK() * model->GetNumField());
    kv_v_->Wait(kv_v_->Pull(feature_ids, &(*v_pull)));
  }
  LOG(INFO) << "got params" << std::endl;
  for (int i = 0; i < gradient_pull->size(); ++i) {
    index_t idx = feature_ids[i];
    real_t weight = (*gradient_pull)[i];
    weight_map[idx] = weight;
    gradient_push_map[idx] = 0.0;
  }
  LOG(INFO) << "model NumK=" << model->GetNumK() << std::endl;
  LOG(INFO) << "val size=" << v_pull->size() << std::endl;
  int num_field = model->GetScoreFunction().compare("ffm") == 0
                  ? model->GetNumField()
                  : 1;
  int num_k = model->GetScoreFunction().compare("linear") == 0
              ? 0
              : model->GetNumK();
  int len = num_field * num_k;
  for (int i = 0; i < feature_ids.size(); ++i) {
    index_t idx = feature_ids[i];
    std::vector<real_t> vec_k;
    for(int j = 0; j < len; ++j) {
      vec_k.push_back((*v_pull)[i * len + j]);
    }
    v_map[idx] = vec_k;
    v_push_map[idx] = std::vector<real_t>(len, 0.0);
  }
  ce_gradient_thread(matrix, model, weight_map, v_map, dist_score_func,
                     is_norm, sum, gradient_push_map, v_push_map, 0, matrix->row_length);
  gradient_push->resize(feature_ids.size());
  for (int i = 0; i < feature_ids.size(); ++i) {
    index_t idx = feature_ids[i];
    real_t g = gradient_push_map[idx];
    (*gradient_push)[i] = g;
  }
  LOG(INFO) << "push params" << std::endl;
  kv_w_->Wait(kv_w_->Push(feature_ids, *gradient_push));
  LOG(INFO) << "finish push w" << std::endl;
  v_push->resize(feature_ids.size() * len);
  for (int i = 0; i < feature_ids.size(); ++i) {
    index_t idx = feature_ids[i];
    for (int j = 0; j < v_push_map[idx].size(); ++j) {
      (*v_push)[i * len + j] = v_push_map[idx][j];
    }
  }
  if (model->GetScoreFunction().compare("fm") == 0 ||
      model->GetScoreFunction().compare("ffm") == 0) {
    kv_v_->Wait(kv_v_->Push(feature_ids, *v_push));
  }
  LOG(INFO) << "finish push v" << std::endl;
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
void DistCrossEntropyLoss::CalcGrad(DMatrix* matrix,
                                    Model& model) {
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_length, 0);
  size_t row_len = matrix->row_length;
  total_example_ += row_len;
  int count = std::ceil(matrix->row_length * 1.0 / batch_size_);
  std::vector<real_t > sum(count, 0.0);
  std::vector<DMatrix> mini_batch_list(count);
  for(int i = 0; i < count; ++ i) {
    // Get a mini-batch from current data matrix
    DMatrix& mini_batch = mini_batch_list[i];
    mini_batch.ResetMatrix(batch_size_);
    index_t len = matrix->GetMiniBatch(batch_size_, mini_batch);
    if (len == 0) {
      break;
    }
    mini_batch.row_length = len;
    // Pull the model parameter from parameter server
    // Calculate gradient
    // Push gradient to the parameter server
    int ti = i;
    pool_->enqueue(std::bind(mini_batch_gradient_thread,
                             &mini_batch,
                             &model,
                             this->dist_score_func_,
                             this->norm_,
                             &(sum[ti]),
                             ti));
  }
  pool_->Sync(count);
 // Accumulate loss
  for (int i = 0; i < sum.size(); ++i) {
    loss_sum_ += sum[i];
  }
}

} // namespace xLearn
