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
This file is the implementation of the Solver class.
*/

#include "src/distributed/dist_solver.h"

#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <thread>

#include "src/base/stringprintf.h"
#include "src/base/split_string.h"
#include "src/base/timer.h"
#include "src/base/system.h"

namespace xLearn {

//------------------------------------------------------------------------------
//         _
//        | |
//   __  _| |     ___  __ _ _ __ _ __
//   \ \/ / |    / _ \/ _` | '__| '_ \
//    >  <| |___|  __/ (_| | |  | | | |
//   /_/\_\______\___|\__,_|_|  |_| |_|
//
//      xLearn   -- 0.10 Version --
//------------------------------------------------------------------------------
void DistSolver::print_logo() const {
  std::string logo = 
"----------------------------------------------------------------------------------------------\n"
                    "           _\n"
                    "          | |\n"
                    "     __  _| |     ___  __ _ _ __ _ __\n"
                    "     \\ \\/ / |    / _ \\/ _` | '__| '_ \\ \n"
                    "      >  <| |___|  __/ (_| | |  | | | |\n"
                    "     /_/\\_\\_____/\\___|\\__,_|_|  |_| |_|\n\n"
                    "        xLearn   -- 0.20 Version --\n"
"----------------------------------------------------------------------------------------------\n"
"\n";
  Color::Modifier green(Color::FG_GREEN);
  Color::Modifier def(Color::FG_DEFAULT);
  Color::Modifier bold(Color::BOLD);
  Color::Modifier reset(Color::RESET);
  std::cout << green << bold << logo << def << reset;
}

/******************************************************************************
 * Creater functions                                                          *
 ******************************************************************************/

// Create Reader by a given string
Reader* DistSolver::create_reader() {
  Reader* reader;
  std::string str = hyper_param_.on_disk ? "disk" : "memory";
  reader = CREATE_READER(str.c_str());
  if (reader == nullptr) {
    LOG(FATAL) << "Cannot create reader: " << str;
  }
  return reader;
}

// Create Score by a given string
DistScore* DistSolver::create_score() {
  DistScore* dist_score;
  dist_score = CREATE_DIST_SCORE(hyper_param_.score_func.c_str());
  if (dist_score == nullptr) {
    LOG(FATAL) << "Cannot create score: "
               << hyper_param_.score_func;
  }
  return dist_score;
}

// Create Loss by a given string
DistLoss* DistSolver::create_loss() {
  DistLoss* loss;
  loss = CREATE_DIST_LOSS(hyper_param_.loss_func.c_str());
  if (loss == nullptr) {
    LOG(FATAL) << "Cannot create loss: "
               << hyper_param_.loss_func;
  }
  return loss;
}

// Create Metric by a given string
Metric* DistSolver::create_metric() {
  Metric* metric;
  metric = CREATE_METRIC(hyper_param_.metric.c_str());
  // Note that here we do not cheack metric == nullptr
  // this is because we can set metric to "none", which 
  // means that we don't print any metric info.
  return metric;
}


/******************************************************************************
 * Functions for xlearn initialize                                            *
 ******************************************************************************/

// Initialize Solver
void DistSolver::Initialize(int argc, char* argv[]) {
  //  Print logo
  print_logo();
  // Check and parse command line arguments
  checker(argc, argv);
  // Initialize log file
  init_log();
  // Init train or predict
  if (hyper_param_.is_train) {
    init_train();
  } else {
    init_predict();
  }
}

// Initialize the xLearn environment through the
// given hyper-parameters. This function will be 
// used for python API.
void DistSolver::Initialize(HyperParam& hyper_param) {
  // Print logo
  print_logo();
  // Check the arguments
  checker(hyper_param);
  this->hyper_param_ = hyper_param;
  // Initialize log file
  init_log();
  // Init train or predict
  if (hyper_param_.is_train) {
    init_train();
  } else {
    init_predict();
  }
}

// Check and parse command line arguments
void DistSolver::checker(int argc, char* argv[]) {
  try {
    dist_checker_.Initialize(hyper_param_.is_train, argc, argv);
    if (!dist_checker_.check_cmd(hyper_param_)) {
      print_error("Arguments error");
      exit(0);
    }
  } catch (std::invalid_argument &e) {
    printf("%s\n", e.what());
    exit(1);
  }
}

// Check the given hyper-parameters
void DistSolver::checker(HyperParam& hyper_param) {
  if (!dist_checker_.check_param(hyper_param)) {
    print_error("Arguments error");
    exit(0);
  }
}

// Initialize log file
void DistSolver::init_log() {
  std::string prefix = get_log_file(hyper_param_.log_file);
  if (hyper_param_.is_train) {
    prefix += "_train";
  } else {
    prefix += "_predict";
  }
  InitializeLogger(StringPrintf("%s.INFO", prefix.c_str()),
              StringPrintf("%s.WARN", prefix.c_str()),
              StringPrintf("%s.ERROR", prefix.c_str()));
}

// Initialize training task
void DistSolver::init_train() {
  /*********************************************************
   *  Initialize thread pool                               *
   *********************************************************/
  size_t threadNumber = std::thread::hardware_concurrency();;
  if (hyper_param_.thread_number != 0) {
    threadNumber = hyper_param_.thread_number;
  }
  LOG(INFO) << "ThreadNumber=" << threadNumber << std::endl;
  pool_ = new ThreadPool(threadNumber);
  /*********************************************************
   *  Initialize Reader                                    *
   *********************************************************/
  Timer timer;
  timer.tic();
  print_action("Read Problem ...");
  LOG(INFO) << "Start to init Reader";
  // Split file
  if (hyper_param_.cross_validation) {
    CHECK_GT(hyper_param_.num_folds, 0);
    splitor_.split(hyper_param_.train_set_file,
                   hyper_param_.num_folds);
    LOG(INFO) << "Split file into "
              << hyper_param_.num_folds
              << " parts.";
  }
  // Get the Reader list
  int num_reader = 0;
  std::vector<std::string> file_list;
  if (hyper_param_.cross_validation) {
    num_reader += hyper_param_.num_folds;
    for (int i = 0; i < hyper_param_.num_folds; ++i) {
      std::string filename = StringPrintf("%s_%d",
           hyper_param_.train_set_file.c_str(), i);
      file_list.push_back(filename);
    }
  } else {  // do not use cross-validation
    num_reader += 1;  // training file
    CHECK_NE(hyper_param_.train_set_file.empty(), true);
    file_list.push_back(hyper_param_.train_set_file);
    if (!hyper_param_.validate_set_file.empty()) {
      num_reader += 1;  // validation file
      file_list.push_back(hyper_param_.validate_set_file);
    }
  }
  LOG(INFO) << "Number of Reader: " << num_reader;
  reader_.resize(num_reader, nullptr);
  // Create Reader
  for (int i = 0; i < num_reader; ++i) {
    reader_[i] = create_reader();
    reader_[i]->Initialize(file_list[i]);
    if (!hyper_param_.on_disk) {
      reader_[i]->SetShuffle(true);
    }
    if (reader_[i] == nullptr) {
      print_error(
        StringPrintf("Cannot open the file %s",
             file_list[i].c_str())
      );
      exit(0);
    }
    LOG(INFO) << "Init Reader: " << file_list[i];
  }
  /*********************************************************
   *  Read problem                                         *
   *********************************************************/
  DMatrix* matrix = nullptr;
  index_t max_feat = 0, max_field = 0;
  for (int i = 0; i < num_reader; ++i) {
    while(reader_[i]->Samples(matrix)) {
      int tmp = matrix->MaxFeat();
      if (tmp > max_feat) { max_feat = tmp; }
      if (hyper_param_.score_func.compare("ffm") == 0) {
        tmp = matrix->MaxField();
        if (tmp > max_field) { max_field = tmp; }
      }
    }
    // Return to the begining of target file.
    reader_[i]->Reset();
  }
  hyper_param_.num_feature = max_feat + 1;
  LOG(INFO) << "Number of feature: " << hyper_param_.num_feature;
  print_info(
    StringPrintf("Number of Feature: %d", 
                 hyper_param_.num_feature)
  );
  if (hyper_param_.score_func.compare("ffm") == 0) {
    hyper_param_.num_field = max_field + 1;
    LOG(INFO) << "Number of field: " << hyper_param_.num_field;
    print_info(
      StringPrintf("Number of Field: %d", 
        hyper_param_.num_field)
    );
  }
  print_info(
    StringPrintf("Time cost for reading problem: %.2f (sec)",
         timer.toc())
  );
  /*********************************************************
   *  Initialize Model                                     *
   *********************************************************/
  timer.reset();
  timer.tic();
  print_action("Initialize model ...");
  // Initialize parameters
  model_ = new Model();
  if (hyper_param_.opt_type.compare("sgd") == 0) {
    hyper_param_.auxiliary_size = 1;
  } else if (hyper_param_.opt_type.compare("adagrad") == 0) {
    hyper_param_.auxiliary_size = 2;
  } else if (hyper_param_.opt_type.compare("ftrl") == 0) {
    hyper_param_.auxiliary_size = 3;
  }
  model_->Initialize(hyper_param_.score_func,
                   hyper_param_.loss_func,
                   hyper_param_.num_feature,
                   hyper_param_.num_field,
                   hyper_param_.num_K,
                   hyper_param_.auxiliary_size,
                   hyper_param_.model_scale);
  index_t num_param = model_->GetNumParameter();
  hyper_param_.num_param = num_param;
  LOG(INFO) << "Number parameters: " << num_param;
  print_info(
    StringPrintf("Model size: %s", 
         PrintSize(num_param*sizeof(real_t)).c_str())
  );
  print_info(
    StringPrintf("Time cost for model initial: %.2f (sec)",
         timer.toc())
  );
  /*********************************************************
   *  Initialize score function                            *
   *********************************************************/
  dist_score_ = create_score();
  dist_score_->Initialize(hyper_param_.learning_rate,
                     hyper_param_.regu_lambda,
                     hyper_param_.alpha,
                     hyper_param_.beta,
                     hyper_param_.lambda_1,
                     hyper_param_.lambda_2,
                     hyper_param_.opt_type);
  LOG(INFO) << "Initialize score function.";
  /*********************************************************
   *  Initialize loss function                             *
   *********************************************************/
  dist_loss_ = create_loss();
  dist_loss_->DistInitialize(dist_score_, pool_, hyper_param_.batch_size,
         hyper_param_.norm, 
         hyper_param_.lock_free);
  LOG(INFO) << "Initialize loss function.";
  /*********************************************************
   *  Init metric                                          *
   *********************************************************/
  metric_ = create_metric();
  if (metric_ != nullptr) {
    metric_->Initialize(pool_);
  }
  LOG(INFO) << "Initialize evaluation metric.";
}

// Initialize predict task
void DistSolver::init_predict() {
  /*********************************************************
   *  Initialize thread pool                               *
   *********************************************************/
  size_t threadNumber = std::thread::hardware_concurrency();
  pool_ = new ThreadPool(threadNumber);
  /*********************************************************
   *  Read model file                                      *
   *********************************************************/
  print_action("Load model ...");
  CHECK_NE(hyper_param_.model_file.empty(), true);
  print_info(
    StringPrintf("Load model from %s",
          hyper_param_.model_file.c_str())
  );
  Timer timer;
  timer.tic();
  model_ = new Model(hyper_param_.model_file);
  hyper_param_.score_func = model_->GetScoreFunction();
  hyper_param_.loss_func = model_->GetLossFunction();
  hyper_param_.num_feature = model_->GetNumFeature();
  if (hyper_param_.score_func.compare("fm") == 0 ||
       hyper_param_.score_func.compare("ffm") == 0) {
    hyper_param_.num_K = model_->GetNumK();
  }
  if (hyper_param_.score_func.compare("ffm") == 0) {
    hyper_param_.num_field = model_->GetNumField();
  }
  print_info(
    StringPrintf("Loss function: %s", 
      hyper_param_.loss_func.c_str())
  );
  print_info(
    StringPrintf("Score function: %s", 
      hyper_param_.score_func.c_str())
  );
  print_info(
    StringPrintf("Number of Feature: %d", 
                 hyper_param_.num_feature)
  );
  if (hyper_param_.score_func.compare("fm") == 0 ||
      hyper_param_.score_func.compare("ffm") == 0) {
    print_info(
      StringPrintf("Number of K: %d", 
                   hyper_param_.num_K)
    );
    if (hyper_param_.score_func.compare("ffm") == 0) {
      print_info(
        StringPrintf("Number of field: %d", 
                    hyper_param_.num_field)
      );
    }
  }
  print_info(
    StringPrintf("Time cost for loading model: %.2f (sec)",
        timer.toc())
  );
  LOG(INFO) << "Initialize model.";
  /*********************************************************
   *  Initialize Reader and read problem                   *
   *********************************************************/
  print_action("Read Problem ...");
  timer.reset();
  timer.tic();
  // Create Reader
  reader_.resize(1, create_reader());
  CHECK_NE(hyper_param_.test_set_file.empty(), true);
  reader_[0]->Initialize(hyper_param_.test_set_file);
  reader_[0]->SetShuffle(false);
  if (reader_[0] == nullptr) {
   print_info(
    StringPrintf("Cannot open the file %s",
                 hyper_param_.test_set_file.c_str())
   );
   exit(0);
  }
  print_info(
    StringPrintf("Time cost for reading problem: %.2f (sec)",
                  timer.toc())
  );
  LOG(INFO) << "Initialize Reader: " << hyper_param_.test_set_file;
  /*********************************************************
   *  Init score function                                  *
   *********************************************************/
  dist_score_ = create_score();
  LOG(INFO) << "Initialize score function.";
  /*********************************************************
   *  Init loss function                                   *
   *********************************************************/
  dist_loss_ = create_loss();
  dist_loss_->DistInitialize(dist_score_, pool_, hyper_param_.norm);
  LOG(INFO) << "Initialize score function.";
}

/******************************************************************************
 * Functions for xlearn start work                                            *
 ******************************************************************************/

// Start training or inference
void DistSolver::StartWork() {
  ps::Postoffice::Get()->SetServerKeyRanges(hyper_param_.num_feature + 1);
  if (hyper_param_.is_train) {
    LOG(INFO) << "Start training work.";
    start_train_work();
  } else {
    LOG(INFO) << "Start inference work.";
    start_prediction_work();
  }
}

// Train
void DistSolver::start_train_work() {
  int epoch = hyper_param_.num_epoch;
  bool early_stop = hyper_param_.early_stop &&
                   !hyper_param_.cross_validation;
  bool quiet = hyper_param_.quiet &&
              !hyper_param_.cross_validation;
  bool save_model = true;
  bool save_txt_model = true;
  if (hyper_param_.model_file.compare("none") == 0 ||
      hyper_param_.cross_validation) {
    save_model = false;
  }
  if (hyper_param_.txt_model_file.compare("none") == 0 ||
      hyper_param_.cross_validation) {
    save_txt_model = false;
  }
  DistTrainer trainer;
  trainer.Initialize(reader_,  /* Reader list */
                     epoch,
                     model_,
                     dist_loss_,
                     metric_,
                     early_stop,
                     quiet);
  print_action("Start to train ...");
/******************************************************************************
 * Training under cross-validation                                            *
 ******************************************************************************/
  if (hyper_param_.cross_validation) {
    trainer.CVTrain();
    print_action("Finish Cross-Validation");
  } 
/******************************************************************************
 * Original training without cross-validation                                 *
 ******************************************************************************/
  else {
    trainer.Train();
    if (save_model) {
      Timer timer;
      timer.tic();
      print_action("Start to save model ...");
      trainer.SaveModel(hyper_param_.model_file);
      print_info(
        StringPrintf("Model file: %s", 
          hyper_param_.model_file.c_str())
      );
      print_info(
        StringPrintf("Time cost for saving model: %.2f (sec)",
             timer.toc())
      );
    } 
    if (save_txt_model) {
      Timer timer;
      timer.tic();
      print_action("Start to save txt model ...");
      trainer.SaveTxtModel(hyper_param_.txt_model_file);
      print_info(
        StringPrintf("TXT Model file: %s", 
          hyper_param_.txt_model_file.c_str())
      );
      print_info(
        StringPrintf("Time cost for saving txt model: %.2f (sec)",
             timer.toc())
      );
    }
    print_action("Finish training");
  }
}

// Inference
void DistSolver::start_prediction_work() {
  print_action("Start to predict ...");
  DistPredictor pdc;
  pdc.Initialize(reader_[0],
                 model_,
                 dist_loss_,
                 hyper_param_.output_file,
                 hyper_param_.sign,
                 hyper_param_.sigmoid);
  // Predict and write output
  pdc.Predict();
}

/******************************************************************************
 * Functions for xlearn finalization                                          *
 ******************************************************************************/

// Finalize xLearn
void DistSolver::Clear() {
  LOG(INFO) << "Clear the xLearn environment ...";
  print_action("Clear the xLearn environment ...");
  // Clear model
  delete this->model_;
  // Clear Reader
  for (size_t i = 0; i < this->reader_.size(); ++i) {
    if (reader_[i] != nullptr) {
      delete reader_[i];
    }
  }
  reader_.clear();
}

} // namespace xLearn
