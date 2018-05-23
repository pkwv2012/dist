#include "iostream"
#include "src/base/math.h"
#include "src/data/hyper_parameters.h"
#include "src/data/parameter_initializer.h"
#include "src/distributed/dist_score_function.h"
#include "ps/ps.h"

#include <time.h>

namespace xLearn {
float alpha = 0.001;
float beta = 1.0;
float lambda1 = 0.00001;
float lambda2 = 2.0;

float regu_lambda = 0.1;
float learning_rate = 0.1;

int v_dim = 1;

extern const int kAlign;

/* Index Table
 * index_t(-1 or max_feature_id + 1): bias
 * 0~max_feature_id:
 *  ...:
 *  id : weight, [latent vector, ]
 *               for FM
 *               [latent vector, ]
 *               for FFM
 *               [latent vector * field, ]
 *  ...:
 * */

static void InitializeFMLatent(real_t* param, int num_K, real_t scale) {
  std::default_random_engine generator;
  std::uniform_real_distribution<real_t> dis(0.0, 1.0);
  real_t coef = 1.0f / sqrt(num_K) * scale;
  for (index_t d = 0; d < num_K; ++ d, ++ param) {
    *param = coef * dis(generator);
  }
}

template <class V>
class KVServerSGDHandle {
public:
  KVServerSGDHandle(const xLearn::HyperParam* hyper_param, bool is_weight)
    : is_weight_(is_weight) {
    hyper_param_ = hyper_param;
  }

public:
  void operator() (const ps::KVMeta& req_meta,
                   const ps::KVPairs<V>& req_data,
                   ps::KVServer<V>* server) {
    auto customer_id = server->get_customer()->customer_id();
    LOG(INFO) << "SGDHandler" << std::endl;
    size_t keys_size = req_data.keys.size();
    ps::KVPairs<V> res;
    int num_K = is_weight_ ? 1 : hyper_param_->num_K;
    int num_field = is_weight_ ? 1 : hyper_param_->num_field;
    int k_aligned = is_weight_ ? 1 : ceil((real_t)num_K / kAlign) * kAlign;
    int len = k_aligned * num_field;
    if (hyper_param_->score_func.compare("linear") == 0) {
      CHECK_EQ(num_K, 1);
      CHECK_EQ(num_field, 1);
    } else if (hyper_param_->score_func.compare("fm") == 0) {
      CHECK_EQ(num_field, 1);
    }
    if (req_meta.push) {
      CHECK_EQ(keys_size * len, req_data.vals.size());
    } else {
      res.keys = req_data.keys;
      res.vals.resize(keys_size * len);
    }
    LOG(INFO) << "keys_size=" << keys_size << std::endl;
    LOG(INFO) << "len=" << len << std::endl;
    for (size_t i = 0; i < keys_size; ++i) {
      ps::Key key = req_data.keys[i];
      if (store_.find(key) == store_.end()) {
        store_[key] = std::vector<V>(len, 0.0);
        if (!is_weight_ && hyper_param_->score_func.compare("fm") == 0) {
          LOG(INFO) << "num_K=" << hyper_param_->num_K << "||scale=" << hyper_param_->model_scale << std::endl;
          ParameterInitializer::Get()->InitializeLatentFactor(store_[key].data(), hyper_param_->auxiliary_size,
                                     hyper_param_->score_func, hyper_param_->num_K, hyper_param_->num_field,
                                     hyper_param_->model_scale);
        }
      }
      std::vector<V>& val = store_[key];
      if (req_meta.push) {
        for (int j = 0; j < val.size(); ++j) {
          float gradient = req_data.vals[i * len + j];
          //gradient += regu_lambda * gradient;
          val[j] += learning_rate * gradient;
        }
      } else {
        for (int j = 0; j < val.size(); ++j) {
          res.vals[i * len + j] = val[j];
        }
      }
    }
    LOG(INFO) << customer_id << "   Responsing" << std::endl;
    // res.DebugPrint(customer_id);
    server->Response(req_meta, res);
    // res.DebugPrint(customer_id);
    LOG(INFO) << customer_id << "    Response finish" << std::endl;
  }
 private:
  std::unordered_map<ps::Key, std::vector<V>> store_;
  // is weight or latent vector
  bool is_weight_;
  const xLearn::HyperParam* hyper_param_;
};

typedef struct AdaGradEntry {
  AdaGradEntry(size_t k = v_dim) {
    w.resize(k, 0.0);
    n.resize(k, 0.0);
  }
  std::vector<float> w;
  std::vector<float> n;
} adagradentry;

struct KVServerAdaGradHandle {
  void operator() (const ps::KVMeta& req_meta,
                   const ps::KVPairs<float>& req_data,
                   ps::KVServer<float>* server) {
    LOG(INFO) << "AdaGradHandler";
    size_t keys_size = req_data.keys.size();
    ps::KVPairs<float> res;
    if (req_meta.push) {
      CHECK_EQ(keys_size * v_dim, req_data.vals.size());
    } else {
      res.keys = req_data.keys;
      res.vals.resize(keys_size * v_dim);
    }
    for (size_t i = 0; i < keys_size; ++i) {
      ps::Key key = req_data.keys[i];
      AdaGradEntry& val = store_[key];
      if (req_meta.push) {
        for (int j = 0; j < val.w.size(); ++j) {
          float gradient = req_data.vals[i * v_dim + j];
          gradient += regu_lambda * gradient;
          val.n[j] = gradient * gradient;
          val.w[j] -= (learning_rate * gradient * InvSqrt(val.n[j]));
        }
      } else {
        for (int j = 0; j < val.w.size(); ++j) {
          res.vals[i * v_dim + j] = val.w[j];
        }
      }
    }
    server->Response(req_meta, res);
  }
 private:
  std::unordered_map<ps::Key, adagradentry> store_;
};

typedef struct FTRLEntry{
  FTRLEntry(size_t k = v_dim) {
    w.resize(k, 0.0);
    n.resize(k, 0.0);
    z.resize(k, 0.0);
  }
  std::vector<float> w;
  std::vector<float> z;
  std::vector<float> n;
} ftrlentry;

struct KVServerFTRLHandle {
  void operator() (const ps::KVMeta& req_meta,
                   const ps::KVPairs<float>& req_data,
                   ps::KVServer<float>* server) {
    LOG(INFO) << "FTRLHandler";
    size_t keys_size = req_data.keys.size();
    ps::KVPairs<float> res;
    if (req_meta.push) {
      CHECK_EQ(keys_size * v_dim, req_data.vals.size());
    } else {
      res.keys = req_data.keys;
      res.vals.resize(keys_size * v_dim);
    }
    for (size_t i = 0; i < keys_size; ++i) {
      ps::Key key = req_data.keys[i];
      FTRLEntry& val = store_[key];
      for (int j = 0; j < val.w.size(); ++j) {
        if (req_meta.push) {
          float gradient = req_data.vals[i * v_dim + j];
          float old_n = val.n[j];
          float n = old_n + gradient * gradient;
          val.z[j] += gradient - (std::sqrt(n) - std::sqrt(old_n)) / alpha * val.w[j];
          val.n[j] = n;
          if (std::abs(val.z[j]) <= lambda1) {
            val.w[j] = 0.0;
          } else {
            float tmpr= 0.0;
            if (val.z[j] > 0.0) tmpr = val.z[j] - lambda1;
            if (val.z[j] < 0.0) tmpr = val.z[j] + lambda1;
            float tmpl = -1 * ( (beta + std::sqrt(val.n[j]))/alpha  + lambda2 );
            val.w[j] = tmpr / tmpl;
          }
        } else {
          res.vals[i * v_dim + j] = val.w[j];
        }
      }
    }
    server->Response(req_meta, res);
  }
 private:
  std::unordered_map<ps::Key, ftrlentry> store_;
};

template<class V>
class XLearnServer{
 public:
  XLearnServer(int argc, char* argv[]){
    kv_w_ = new ps::KVServer<V>(0);
    kv_v_ = new ps::KVServer<V>(1);
    checker_ = new xLearn::Checker;
    checker_->Initialize(hyper_param_.is_train, argc, argv);
    checker_->check_cmd(hyper_param_);

    alpha = hyper_param_.alpha;
    beta = hyper_param_.beta;
    lambda1 = hyper_param_.lambda_1;
    lambda2 = hyper_param_.lambda_2;
    regu_lambda = hyper_param_.regu_lambda;
    learning_rate = hyper_param_.learning_rate;

    if (hyper_param_.score_func.compare("linear") == 0) {
      v_dim = 1;
    }
    if (hyper_param_.score_func.compare("fm")  == 0 ||
        hyper_param_.score_func.compare("ffm") == 0) {
      v_dim = hyper_param_.num_K;
    }

    if (hyper_param_.opt_type.compare("sgd") == 0) {
      kv_w_->set_request_handle(KVServerSGDHandle<V>(&hyper_param_, true));
      kv_v_->set_request_handle(KVServerSGDHandle<V>(&hyper_param_, false));
    }
    if (hyper_param_.opt_type.compare("adagrad") == 0) {
      kv_w_->set_request_handle(KVServerAdaGradHandle());
      kv_v_->set_request_handle(KVServerAdaGradHandle());
    }
    if (hyper_param_.opt_type.compare("ftrl") == 0) {
      kv_w_->set_request_handle(KVServerFTRLHandle());
      kv_v_->set_request_handle(KVServerFTRLHandle());
    }
    std::cout << "init server success " << std::endl;
  }

  ~XLearnServer() {
    delete kv_w_;
    delete kv_v_;
    delete checker_;
  }
  ps::KVServer<V>* kv_w_;
  ps::KVServer<V>* kv_v_;
  xLearn::Checker* checker_;
  xLearn::HyperParam hyper_param_;
};//end class Server
}
