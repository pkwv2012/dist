//
// Created by pkwv on 5/7/18.
//


#include <string>
#include "gtest/gtest.h"

#include "src/base/common.h"
#include "src/distributed/dist_score_function.h"
#include "src/distributed/dist_fm_score.h"


namespace xLearn {

TEST(DistFMScoreTest, CreateScore) {
  DistScore* score = CREATE_DIST_SCORE("fm");
  score->GetOptType();
}// Init hyper-parameters

HyperParam Init() {
  HyperParam hyper_param;
  hyper_param.score_func = "fm";
  hyper_param.loss_func = "squared";
  hyper_param.num_feature = 10;
  hyper_param.num_K = 8;
  hyper_param.auxiliary_size = 2;
  hyper_param.model_file = "./test_model.bin";
  return hyper_param;
}

TEST(DistFMScoreTest, CalcGradient) {
  DMatrix matrix;
  matrix.ResetMatrix(10);
  HyperParam hyper_param = Init();
  Model model_fm;
  model_fm.Initialize(hyper_param.score_func,
                    hyper_param.loss_func,
                    hyper_param.num_feature,
                    hyper_param.num_field,
                    hyper_param.num_K,
                    hyper_param.auxiliary_size);
  std::unordered_map<index_t, real_t> w;
  std::unordered_map<index_t, std::vector<real_t>> v;
  real_t* sum = new real_t[10];
  std::unordered_map<index_t, real_t> w_g;
  std::unordered_map<index_t, std::vector<real_t>> v_g;
  for (index_t i = 0; i < 10; ++ i) {
    v[i] = std::vector<real_t>(8);
    v_g[i] = std::vector<real_t>(8);
  }

  size_t start_idx(0);
  size_t end_idx(9);
  DistScore* dist_score_func = CREATE_DIST_SCORE("fm");
  dist_score_func->Initialize(hyper_param.learning_rate,
                          hyper_param.regu_lambda,
                          hyper_param.alpha,
                          hyper_param.beta,
                          hyper_param.lambda_1,
                          hyper_param.lambda_2,
                          std::string("sgd"));
  CHECK_GE(end_idx, start_idx);
  CHECK_NOTNULL(dist_score_func);
  LOG(INFO) << dist_score_func->GetOptType() << std::endl;
  dist_score_func->DistCalcGrad(&matrix, model_fm, w, v, sum, w_g, v_g, start_idx, end_idx);
}

} // namespace xLearn

