//
// Created by pkwv on 5/27/18.
//

#ifndef XLEARN_SERVER_HANDLER_H
#define XLEARN_SERVER_HANDLER_H

#include <ps-lite/include/ps/kv_app.h>
#include <src/data/hyper_parameters.h>

namespace xLearn {

template <class V>
class ServerHandler {
 public:
  ServerHandler(const HyperParam* hyper_param,
                const int& num_K)
    : num_K_(num_K){
    hyper_param_ = hyper_param;
  }

  virtual void operator() (const ps::KVMeta& req_meta,
                           const ps::KVPairs<V>& req_data,
                           ps::KVServer<V>* server) {
    size_t keys_size = req_data.keys.size();
    ps::KVPairs<V> res;
    int num_field = hyper_param_->num_field;
    int k_aligned = ceil((real_t)num_K_ / kAlign) * kAlign;
    int len = k_aligned * num_field;
  }

  virtual ~ServerHandler() {}

 private:
  std::unordered_map<ps::Key, std::vector<V>> store_;
  const HyperParam* hyper_param_;
  const int num_K_;
};

template <class V>
class ServerSGDHandler: public ServerHandler<V> {
 public:
  ServerSGDHandler(const HyperParam* hyper_param,
                   const int& num_K)
    : ServerHandler(hyper_param, num_K) {
  }

  virtual void operator() (const ps::KVMeta& req_meta,
                           const ps::KVPairs<V>& req_data,
                           ps::KVServer<V>* server) {

  }
};

}

#endif //XLEARN_SERVER_HANDLER_H
