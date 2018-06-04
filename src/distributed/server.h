#include "iostream"
#include "src/base/math.h"
#include "src/data/hyper_parameters.h"
#include "src/data/parameter_initializer.h"
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

// `kAlign` can be will designed.
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

template <class V>
class KVServerSGDHandle {
public:
  KVServerSGDHandle(const xLearn::HyperParam* hyper_param, bool is_weight)
    : is_weight_(is_weight), decay_speed_(hyper_param->decay_speed) {
    hyper_param_ = hyper_param;
    mini_batch_count_ = 0;
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
      CHECK_EQ(len, 1);
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
        if (!is_weight_
            && (hyper_param_->score_func.compare("fm") == 0
                || hyper_param_->score_func.compare("ffm") == 0)) {
          LOG(INFO) << "num_K=" << hyper_param_->num_K << "||scale=" << hyper_param_->model_scale << std::endl;
          ParameterInitializer::Get()->InitializeLatentFactor(store_[key].data(), hyper_param_->auxiliary_size,
                                     hyper_param_->score_func, hyper_param_->num_K, hyper_param_->num_field,
                                     hyper_param_->model_scale);
        }
      }
      std::vector<V>& val = store_[key];
      if (req_meta.push) {
        int count = mini_batch_count_ / 10000;
        real_t decay_rate = decay_speed_ < 0.0 ? 1.0 : sqrt(decay_speed_ / (count + decay_speed_));
        for (int j = 0; j < val.size(); ++j) {
          float gradient = req_data.vals[i * len + j];
          //gradient += regu_lambda * gradient;
          val[j] += learning_rate * gradient * decay_rate;
        }
      } else {
        for (int j = 0; j < val.size(); ++j) {
          res.vals[i * len + j] = val[j];
        }
      }
    }
    ++ mini_batch_count_;
    LOG(INFO) << customer_id << "   Responsing" << std::endl;
    if (req_meta.push) {
      //receive_count_ ++;
    }
    server->Response(req_meta, res);
    LOG(INFO) << customer_id << "    Response finish" << std::endl;
  }
 private:
  int id(const ps::KVMeta& req_meta) {
    // TODO
    return -1;
  }
  std::unordered_map<ps::Key, std::vector<V>> store_;
  // is weight or latent vector
  bool is_weight_;
  const xLearn::HyperParam* hyper_param_;
  int mini_batch_count_;
  //std::condition_variable receive_barrier_;
  //std::atomic<int> receive_count_;
  const double decay_speed_;
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
      CHECK(0) << "Adagrad not declared" << std::endl;
      //kv_w_->set_request_handle(KVServerAdaGradHandle());
      //kv_v_->set_request_handle(KVServerAdaGradHandle());
    }
    if (hyper_param_.opt_type.compare("ftrl") == 0) {
      CHECK(0) << "Ftrl not declared" << std::endl;
      //kv_w_->set_request_handle(KVServerFTRLHandle());
      //kv_v_->set_request_handle(KVServerFTRLHandle());
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
