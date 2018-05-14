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

This file is the entry for training of the xLearn.
*/

#include "src/base/common.h"
#include "src/base/timer.h"
#include "src/base/stringprintf.h"
#include "src/distributed/dist_solver.h"
#include "src/distributed/server.h"
#include "src/distributed/worker.h"

//------------------------------------------------------------------------------
// The pre-defined main function
//------------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  ps::Start(0);
  if (ps::IsServer()) {
    xlearn::XLearnServer* server = new xlearn::XLearnServer(argc, argv);
  }

  if (ps::IsWorker()) {
    Timer timer;
    timer.tic();
    xLearn::DistSolver dist_solver;
    dist_solver.SetTrain();
    dist_solver.Initialize(argc, argv);
    dist_solver.StartWork();
    dist_solver.Clear();
    print_info(
      StringPrintf("Total time cost: %.2f (sec)", 
      timer.toc()), false);
  }
  ps::Finalize(0);
  return 0;
}
