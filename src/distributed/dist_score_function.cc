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
This file is the implementation of the base Score class.
*/

#include "src/distributed/dist_score_function.h"
#include "src/distributed/dist_linear_score.h"
#include "src/distributed/dist_fm_score.h"
#include "src/distributed/dist_ffm_score.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(xLearn_dist_score_registry, DistScore);
REGISTER_SCORE("dist_linear", DistLinearScore);
REGISTER_SCORE("dist_fm", DistFMScore);
REGISTER_SCORE("dist_ffm", DistFFMScore);

}  // namespace xLearn
