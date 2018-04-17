#include "iostream"
#include "src/base/math.h"
#include "src/data/hyper_parameters.h"
#include "src/distributed/dist_checker.h"
#include "src/distributed/dist_score_function.h"
#include "ps/ps.h"

#include <time.h>

namespace xlearn {
float alpha = 0.001;
float beta = 1.0;
float lambda1 = 0.00001;
float lambda2 = 2.0;

float regu_lambda = 0.1;
float learning_rate = 0.1;

int v_dim = 1;

typedef struct SGDEntry{
  SGDEntry(size_t k = v_dim) {
    w.resize(k, 0.0);
  }
  std::vector<float> w;
} sgdentry;

struct KVServerSGDHandle {
  void operator() (const ps::KVMeta& req_meta,
                   const ps::KVPairs<float>& req_data,
                   ps::KVServer<float>* server) {
    LOG(INFO) << "SGDHandler" << std::endl;
    size_t keys_size = req_data.keys.size();
    ps::KVPairs<float> res;
    if (req_meta.push) {
      CHECK_EQ(keys_size * v_dim, req_data.vals.size());
    } else {
      res.keys = req_data.keys;
      res.vals.resize(keys_size);
    }
    LOG(INFO) << "keys_size=" << keys_size << std::endl;
    for (size_t i = 0; i < keys_size; ++i) {
      ps::Key key = req_data.keys[i];
      SGDEntry& val = store_[key];
      if (req_meta.push) {
        for (int j = 0; j < val.w.size(); ++j) {
          float gradient = req_data.vals[i * v_dim + j];
          gradient += regu_lambda * gradient;
          val.w[j] -= learning_rate * gradient;
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
  std::unordered_map<ps::Key, sgdentry> store_;
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
      res.vals.resize(keys_size);
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
      res.vals.resize(keys_size);
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

class XLearnServer{
 public:
  XLearnServer(int argc, char* argv[]){
    auto server_ = new ps::KVServer<float>(0);
    checker_ = new xLearn::DistChecker;
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
      server_->set_request_handle(KVServerSGDHandle());
    }
    if (hyper_param_.opt_type.compare("adagrad") == 0) {
      server_->set_request_handle(KVServerAdaGradHandle());
    }
    if (hyper_param_.opt_type.compare("ftrl") == 0) {
      server_->set_request_handle(KVServerFTRLHandle());
    }
    std::cout << "init server success " << std::endl;
  }
  ~XLearnServer(){}
  ps::KVServer<float>* server_;
  xLearn::HyperParam hyper_param_;
  xLearn::DistChecker* checker_;
};//end class Server
}
