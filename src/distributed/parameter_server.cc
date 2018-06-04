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

This file is the implementation of KVStore.
*/

#include "src/distributed/parameter_server.h"

namespace xLearn {

// Push a list of (key, value) into store.
// For example:
//  ------------------------------------------------------
// |  key:   |  0  |  2  |  4  |  5  |  6   |  7   |  9   |
// | value:  | 0.2 | 1.0 | 0.5 | 1.0 | 0.33 |  0.7 |  0.8 |
//  ------------------------------------------------------
void KVStore::Push(const std::vector<index_t>& key,
   	               const std::vector<real_t>& value) {
 // TODO(chao)
}

void KVStore::Push(std::vector<ps::Key> &key, Model &gradient) {
  {
    index_t aux_size = gradient.GetAuxiliarySize();
    real_t* weight = gradient.GetParameter_w();
    for (index_t i = 0; i < gradient.GetAuxiliarySize(); ++ i) {
      *(weight + (key.size() - 1) * aux_size + i) = *(gradient.GetParameter_b() + i);
    }
    ps::SArray<real_t> w(weight, gradient.GetNumParameter_w() + aux_size, false);
    ps::SArray<ps::Key> ps_key(key.data(), key.size(), false);
    CHECK_EQ(ps_key.size() * aux_size, w.size());
    kv_w_.Wait(kv_w_.ZPush(ps::SArray<ps::Key>(ps_key), w));
  }
  ps::SArray<real_t> v(gradient.GetParameter_v(), gradient.GetNumParameter_v(), false);
  if (gradient.GetScoreFunction().compare("fm") == 0
      || gradient.GetScoreFunction().compare("ffm") == 0) {
    ps::SArray<ps::Key > ps_key(key.data(), key.size() - 1, false);
    CHECK_EQ(ps_key.size() * gradient.GetNumK() * gradient.GetNumField() * gradient.GetAuxiliarySize(),
             v.size());
    kv_v_.Wait((kv_v_.ZPush(ps::SArray<ps::Key>(ps_key), v)));
  }
}

// Push a list of (key, value_list) into store.
// For example:
//  ------------------------------------------------------
// |  key:   |  0  |  2  |  4  |  5  |  6   |  7   |  9   |
// | value:  | 0.2 | 1.0 | 0.5 | 1.0 | 0.33 |  0.7 |  0.8 |
// |         | 0.1 | 1.2 | 0.1 | 0.8 | 0.9  |  1.0 |  0.5 |
// |         | 0.5 | 1.4 | 1.7 | 1.5 | 0.8  |  0.7 |  0.6 |
// |         | 0.2 | 1.2 | 1.4 | 1.8 | 0.5  |  1.1 |  1.8 |
// |         | ..  | ..  | ..  | ..  | ..   |  ..  |  ..  |
//  ------------------------------------------------------
// This method is useful for the FM and FFM task.
void KVStore::Push(const std::vector<index_t>& key,
   	               const std::vector<real_t>& value_list,
   	               const size_t length) {
 // TODO(chao)
}

// Pull the values for a list of keys from store.
// For example:
//  ------------------------------------------------------
// |  key:   |  0  |  2  |  4  |  5  |  6   |  7   |  9   |
// | value:  | 0.2 | 1.0 | 0.5 | 1.0 | 0.33 |  0.7 |  0.8 |
//  ------------------------------------------------------
void KVStore::Pull(const std::vector<index_t>& key,
   	               std::vector<real_t>* value) {
  // TODO(chao)
}

// Pull the value list for a list of keys from store.
// For example:
//  ------------------------------------------------------
// |  key:   |  0  |  2  |  4  |  5  |  6   |  7   |  9   |
// | value:  | 0.2 | 1.0 | 0.5 | 1.0 | 0.33 |  0.7 |  0.8 |
// |         | 0.1 | 1.2 | 0.1 | 0.8 | 0.9  |  1.0 |  0.5 |
// |         | 0.5 | 1.4 | 1.7 | 1.5 | 0.8  |  0.7 |  0.6 |
// |         | 0.2 | 1.2 | 1.4 | 1.8 | 0.5  |  1.1 |  1.8 |
// |         | ..  | ..  | ..  | ..  | ..   |  ..  |  ..  |
//  ------------------------------------------------------
// This method is useful for the FM and FFM task.
void KVStore::Pull(const std::vector<index_t>& key,
   	               std::vector<real_t>* value_list,
   	               const size_t length) {
 // TODO(Chao)
}

void KVStore::Pull(std::vector<ps::Key>& key,
                   Model* model) {
  index_t aux_size = model->GetAuxiliarySize();
  ps::SArray<real_t> w(model->GetParameter_w(), key.size(), false);
  {
    ps::SArray<ps::Key> ps_key(key.data(), key.size(), false);
    kv_w_.Wait(kv_w_.ZPull(ps::SArray<ps::Key>(ps_key), &w));
    model->SetParamB(w.data() +((key.size() - 1) * model->GetAuxiliarySize()));
  }
  ps::SArray<real_t> v(model->GetParameter_v(), model->GetNumParameter_v(), false);
  if (model->GetScoreFunction().compare("fm") == 0
      || model->GetScoreFunction().compare("ffm") == 0) {
    ps::SArray<ps::Key> ps_key(key.data(), key.size() - 1, false);
    kv_v_.Wait((kv_v_.ZPull(ps::SArray<ps::Key>(ps_key), &v)));
  }
}

//------------------------------------------------------------------------------
// In xLearn, we use a simple range strategy for model partiton
// on parameter server. For example, we have 10 features and 3 
// server nodes.
//
//  ---------------------------------------
// | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
//  ---------------------------------------
//   |   |   |   |   |   |   |   |   |   |
//  s0  s1  s2  s0  s1  s2  s0  s1  s2  s0
//
// On each local server:
//
//        s0                  s1               s2
//  ---------------      -----------      -----------
// | 0 | 1 | 2 | 3 |    | 0 | 1 | 2 |    | 0 | 1 | 2 |
//  ---------------      -----------      -----------
//   |   |   |   |        |   |   |        |   |   |
//   0   3   6   9        1   4   7        2   5   8
//------------------------------------------------------------------------------

// Given a feature id, return the server id, which stores that feature.
size_t KVStore::GetServerId(const index_t feat_id) const {
  CHECK_GE(feat_id, 0);
  return feat_id % server_num_;
}

// Mapping the global feature id to the local server id.
index_t KVStore::FeatMap(const index_t feat_id) const {
  CHECK_GE(feat_id, 0);
  return feat_id / server_num_;
}

}  // namespace xLearn