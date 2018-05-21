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
  ps::KVWorker<real_t> kv_w(0, thread_id);
  ps::KVWorker<real_t> kv_v(1, thread_id);
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
    //model->SetParamW(param_w.data(), param_w.size() - 1);
    //model->SetParamB(param_w.data() + feat_num);
    //model->SetParamV(param_v.data());
    model->SetParamB(param_w.data() + feat_num);

    for (int j = i; j < i_end_idx; ++ j) {
      SparseRow* row = matrix->row[j];
      real_t norm = is_norm ? matrix->norm[i] : 1.0;
      (*pred)[j] = score_func_->CalcScore(row, *model, norm);
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
    Model& model_i = model_arr[i];
    model_i.Initialize(model.GetScoreFunction(), model.GetLossFunction(),
                       batch_feature_num * 1.2, model.GetNumField(),
                       model.GetNumK(), model.GetAuxiliarySize(),
                       model.GetScale());
    pool_->enqueue(std::bind(pred_thread_mini_batch,
                             matrix,
                             &model_i,
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

}  // namespace xLearn
