/*****************************************************************************/
/*  Copyright (c) 2015, Karl Pauwels                                         */
/*  All rights reserved.                                                     */
/*                                                                           */
/*  Redistribution and use in source and binary forms, with or without       */
/*  modification, are permitted provided that the following conditions       */
/*  are met:                                                                 */
/*                                                                           */
/*  1. Redistributions of source code must retain the above copyright        */
/*  notice, this list of conditions and the following disclaimer.            */
/*                                                                           */
/*  2. Redistributions in binary form must reproduce the above copyright     */
/*  notice, this list of conditions and the following disclaimer in the      */
/*  documentation and/or other materials provided with the distribution.     */
/*                                                                           */
/*  3. Neither the name of the copyright holder nor the names of its         */
/*  contributors may be used to endorse or promote products derived from     */
/*  this software without specific prior written permission.                 */
/*                                                                           */
/*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      */
/*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        */
/*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    */
/*  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     */
/*  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   */
/*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         */
/*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    */
/*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    */
/*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      */
/*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    */
/*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     */
/*****************************************************************************/

#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <math.h>
#include <cstring>
#include <cuda_profiler_api.h>
#include <stdexcept>
#include <Eigen/Dense>
#include <d_multiple_rigid_poses.h>
#include <utility_kernels_pose.h>
#include <multiple_rigid_pose_kernels.h>
#include <utilities.h>

namespace pose {

D_MultipleRigidPoses::D_MultipleRigidPoses(int n_cols, int n_rows,
                                           float nodal_point_x,
                                           float nodal_point_y,
                                           float focal_length_x,
                                           float focal_length_y, float baseline,
                                           Parameters parameters)
    : parameters_{ parameters }, _n_cols{ n_cols }, _n_rows{ n_rows },
      _nodal_point_x{ nodal_point_x }, _nodal_point_y{ nodal_point_y },
      _focal_length_x{ focal_length_x }, _focal_length_y{ focal_length_y },
      _baseline{ baseline }, _currentSparseObject{ 0 },
      _multipleRigidModelsOgre{
        std::unique_ptr<MultipleRigidModelsOgre>{ new MultipleRigidModelsOgre(
            _n_cols, _n_rows, _focal_length_x, _focal_length_y, _nodal_point_x,
            _nodal_point_y, parameters_.near_plane_, parameters_.far_plane_) }
      },
      cub_radix_sorter_{
        std::unique_ptr<util::CubRadixSorter<unsigned int, int> >{
          new util::CubRadixSorter<unsigned int, int>(
              _n_rows * _n_cols * _N_FLOWS, 0, parameters_.getKeyBits())
        }
      },
      _n_objects{ 0 }, _running{ true }, render_state_changed_{ true } {

  camera_pose_.setValid(true);
  previous_camera_pose_.setValid(false);

  /*******************************************************/
  /* Pre-allocate (and initiate )all device space needed */
  /* with maximal possible size                          */
  /* to avoid re-allocation and re-computation           */
  /*******************************************************/

  // linear index to store pixel locations
  // goes from 0 to n_rows*n_cols*n_flows
  // used to identify flow source
  std::vector<int> h_linear_ind(_n_rows * _n_cols * _N_FLOWS);
  std::iota(h_linear_ind.begin(), h_linear_ind.end(), 0);
  d_linear_ind_ =
      util::Device1D<int>::make_unique(_n_rows * _n_cols * _N_FLOWS);
  d_linear_ind_->copyFrom(h_linear_ind);

  // 0,1,2,...,n_objects valid locations
  d_valid_flow_Zbuffer_ =
      util::Device1D<unsigned int>::make_unique(_n_rows * _n_cols * _N_FLOWS);
  d_valid_disparity_Zbuffer_ =
      util::Device1D<unsigned int>::make_unique(_n_rows * _n_cols);
  d_valid_flow_Zbuffer_sub_ =
      util::Device1D<unsigned int>::make_unique(_n_rows * _n_cols * _N_FLOWS);
  d_valid_disparity_Zbuffer_sub_ =
      util::Device1D<unsigned int>::make_unique(_n_rows * _n_cols);
  d_extra_disparity_buffer_ =
      util::Device1D<unsigned int>::make_unique(_n_rows * _n_cols);

  // index of valid locations
  d_ind_flow_Zbuffer_ =
      util::Device1D<int>::make_unique(_n_rows * _n_cols * _N_FLOWS);
  d_ind_disparity_Zbuffer_ =
      util::Device1D<int>::make_unique(_n_rows * _n_cols);
  d_ind_flow_Zbuffer_sub_ =
      util::Device1D<int>::make_unique(_n_rows * _n_cols * _N_FLOWS);
  d_ind_disparity_Zbuffer_sub_ =
      util::Device1D<int>::make_unique(_n_rows * _n_cols);

  // starting index of sorted segments
  d_seg_start_inds_ =
      util::Device1D<int>::make_unique(parameters_.max_objects_);

  // Gathering
  d_flow_compact_ =
      util::Device1D<float2>::make_unique(_n_cols * _n_rows * _N_FLOWS);
  d_Zbuffer_flow_compact_ =
      util::Device1D<float>::make_unique(_n_cols * _n_rows * _N_FLOWS);
  d_disparity_compact_ = util::Device1D<float>::make_unique(_n_cols * _n_rows);
  d_Zbuffer_normals_compact_ =
      util::Device1D<float4>::make_unique(_n_cols * _n_rows);
  d_n_values_flow_ = util::Device1D<int>::make_unique(parameters_.max_objects_);
  d_start_ind_flow_ =
      util::Device1D<int>::make_unique(parameters_.max_objects_);
  d_n_values_disparity_ =
      util::Device1D<int>::make_unique(parameters_.max_objects_);
  d_start_ind_disparity_ =
      util::Device1D<int>::make_unique(parameters_.max_objects_);

  // Normal equations
  d_CO_ = util::Device1D<float>::make_unique(
      4 * _MAX_N_VAL_ACCUM * _N_CON_FLOW * parameters_.max_objects_);
  d_CO_reduced_ = util::Device1D<float>::make_unique(_N_CON_FLOW *
                                                     parameters_.max_objects_);
  d_CD_ = util::Device1D<float>::make_unique(
      4 * _MAX_N_VAL_ACCUM * _N_CON_DISP * parameters_.max_objects_);
  d_CD_reduced_ = util::Device1D<float>::make_unique(_N_CON_DISP *
                                                     parameters_.max_objects_);
  d_abs_res_ =
      util::Device1D<float>::make_unique((_N_FLOWS + 1) * _n_rows * _n_cols);

  h_CO_reduced_.resize(_N_CON_FLOW * parameters_.max_objects_);
  h_CD_reduced_.resize(_N_CON_DISP * parameters_.max_objects_);
  segment_normal_eqs_.resize(parameters_.max_objects_);

  // Segment starting indices in single residual structure
  d_offset_ind_res_flow_ =
      util::Device1D<int>::make_unique(parameters_.max_objects_);
  d_offset_ind_res_disparity_ =
      util::Device1D<int>::make_unique(parameters_.max_objects_);
  d_dTR_ = util::Device1D<float>::make_unique(6 * parameters_.max_objects_);
  d_delta_T_accum_ =
      util::Device1D<float>::make_unique(3 * parameters_.max_objects_);
  d_delta_Rmat_accum_ =
      util::Device1D<float>::make_unique(9 * parameters_.max_objects_);
  d_segment_translation_table_ =
      util::Device1D<int>::make_unique(parameters_.max_objects_);

  // Approximate median
  // pre-generate maximum number of random numbers required (uniform between
  // [0,1[ , used for sampling (with replacement) of absolute residuals
  std::vector<float> h_random_numbers(_n_rows * _n_cols * (_N_FLOWS + 1));
  srand(0);
  for (auto &it : h_random_numbers)
    it = (float)((double)rand() / double(RAND_MAX));
  d_random_numbers_ =
      util::Device1D<float>::make_unique(_n_rows * _n_cols * (_N_FLOWS + 1));
  d_random_numbers_->copyFrom(h_random_numbers);
  int pp = int(ceil(double(_n_rows * _n_cols * (_N_FLOWS + 1)) /
                    243.0)); // second stage reduction
  d_median_tmp_ = util::Device1D<float>::make_unique(pp);
  d_abs_res_scales_ =
      util::Device1D<float>::make_unique(parameters_.max_objects_);

  // temps used inside median calculation
  d_median_n_in_ = util::Device1D<int>::make_unique(parameters_.max_objects_);
  d_median_start_inds_ =
      util::Device1D<int>::make_unique(parameters_.max_objects_);

  // Initial Zbuffer
  d_init_Z_ = util::Device1D<float>::make_unique(_n_rows * _n_cols);

  // Residual flow
  d_res_flowx_ = util::Device1D<float>::make_unique(_n_rows * _n_cols);
  d_res_flowy_ = util::Device1D<float>::make_unique(_n_rows * _n_cols);

  // Residual ar flow
  d_res_ar_flowx_ = util::Device1D<float>::make_unique(_n_rows * _n_cols);
  d_res_ar_flowy_ = util::Device1D<float>::make_unique(_n_rows * _n_cols);
}

void D_MultipleRigidPoses::addModel(const char *obj_filename, float obj_scale,
                                    TranslationRotation3D initial_pose) {
  if (_n_objects > (parameters_.max_objects_ - 2))
    throw std::runtime_error("D_MultipleRigidPoses::addModel: Max objects "
                             "exceeded, increase key_bits\n");

  _n_objects++;
  _currentPoses.push_back(initial_pose);
  setSparsePose(initial_pose, _n_objects - 1);

  _multipleRigidModelsOgre->addModel(obj_filename);
}

void D_MultipleRigidPoses::removeAllModels() {
  _multipleRigidModelsOgre->removeAllModels();
  _currentPoses.clear();
  _currentSparseObject = 0;
  _n_objects = 0;
}

void D_MultipleRigidPoses::update(const util::Device1D<float> &d_flowx,
                                  const util::Device1D<float> &d_flowy,
                                  const util::Device1D<float> &d_ar_flowx,
                                  const util::Device1D<float> &d_ar_flowy,
                                  const util::Device2D<float> &d_disparity,
                                  std::bitset<32> segments_to_update) {
// small inaccuracy here is that the optical flow (frame 1 -> frame 2) is
// segmented according to the region estimated at frame 2
// this is necessary in order to update the residual flow according to the
// updated model depth
// have shown (in simulation) that the resulting pose estimates are a lot more
// accurate than those based on initial depth

#ifdef TIME_STEPS
  // Setup timers
  cudaEvent_t start, end;
  float elapsed_time;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
#endif

  if (_running) {

    // reset timers
    int n_timers = 14;
    _compTimes.clear();
    _compTimes.assign(n_timers, 0.0);

    /* 0 preprocess-total
     * 1 ols-total
     * 2 robust-total
     * 3 mark valids
     * 4 radix sort
     * 5 gather
     * 6 compose normal equations
     * 7 reduce normal equations
     * 8 solve normal equations (cpu)
     * 9 compute absolute residuals
     * 10 median absolute residuals
     * 11 render
     * 12 pose accumulation (cpu)
     * 13 residual flow
     */

    std::vector<TranslationRotation3D> delta_poses_accum(
        _n_objects, TranslationRotation3D());

    std::vector<TranslationRotation3D> explained_delta_poses(
        _n_objects, TranslationRotation3D());

#ifdef TIME_STEPS
    cudaEventRecord(start, 0);
#endif

    // first rendering
    render(_currentPoses);

#ifdef TIME_STEPS
    cudaThreadSynchronize();
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    _compTimes.at(11) += (double)elapsed_time;
#endif

#ifdef TIME_STEPS
    cudaEventRecord(start, 0);
#endif

    // store initial Zbuffer as initial model for residual flow
    convertZbufferToZ(d_init_Z_->data(), _multipleRigidModelsOgre->getZBuffer(),
                      _n_cols, _n_rows, _nodal_point_x, _nodal_point_y,
                      parameters_.near_plane_, parameters_.far_plane_);

#ifdef TIME_STEPS
    cudaThreadSynchronize();
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    _compTimes.at(13) += (double)elapsed_time;
#endif

    // start iterations
    std::vector<float> dT_accum(3 * _n_objects);
    std::vector<float> dR_accum(9 * _n_objects);
    std::vector<TranslationRotation3D> poses_accum = _currentPoses;

    for (int it = 1; it <= parameters_.n_icp_outer_it_; it++) {

// update residual flow
#ifdef TIME_STEPS
      cudaEventRecord(start, 0);
#endif

      for (int o = 0; o < _n_objects; o++) {
        auto delta_pose =
            poses_accum.at(o) * _currentPoses.at(o).inverseTransform();
        if (previous_camera_pose_.isValid()) {
          explained_delta_poses.at(o) = camera_pose_.inverseTransform() *
                                        delta_pose * previous_camera_pose_;
        } else {
          explained_delta_poses.at(o) = delta_pose;
        }
      }

      for (int o = 0; o < _n_objects; o++) {
        explained_delta_poses.at(o).getT(&dT_accum[3 * o]);
        explained_delta_poses.at(o).getR_mat(&dR_accum[9 * o]);
      }
      d_delta_T_accum_->copyFrom(dT_accum, 3 * _n_objects);
      d_delta_Rmat_accum_->copyFrom(dR_accum, 9 * _n_objects);

#ifdef TIME_STEPS
      cudaThreadSynchronize();
      cudaEventRecord(end, 0);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, start, end);
      _compTimes.at(12) += (double)elapsed_time;
#endif

#ifdef TIME_STEPS
      cudaEventRecord(start, 0);
#endif
      computeResidualFlow(
          d_res_flowx_->data(), d_res_flowy_->data(), d_res_ar_flowx_->data(),
          d_res_ar_flowy_->data(), d_flowx.data(), d_flowy.data(),
          d_ar_flowx.data(), d_ar_flowy.data(), d_delta_T_accum_->data(),
          d_delta_Rmat_accum_->data(), d_init_Z_->data(),
          _multipleRigidModelsOgre->getSegmentIND(), _n_cols, _n_rows,
          _nodal_point_x, _nodal_point_y, _focal_length_x, _focal_length_y);

#ifdef TIME_STEPS
      cudaThreadSynchronize();
      cudaEventRecord(end, 0);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, start, end);
      _compTimes.at(13) += (double)elapsed_time;
#endif

      robustPoseUpdates(d_res_flowx_->data(), d_res_flowy_->data(),
                        d_res_ar_flowx_->data(), d_res_ar_flowy_->data(),
                        d_disparity.data(), d_disparity.pitch(),
                        (int)segments_to_update.to_ulong());

      // accumulate delta poses
      for (int s = 0; s < _segment_info.size(); s++) {
        SegmentINFO &tmp = _segment_info.at(s);
        delta_poses_accum.at(tmp.segment_ind) =
            _robustDeltaPoses.at(s) * delta_poses_accum.at(tmp.segment_ind);
      }

#ifdef TIME_STEPS
      cudaEventRecord(start, 0);
#endif

      // update poses, invalidating if magnitude T too large
      poses_accum = _currentPoses;

      // transform to camera frame
      for (auto &it : poses_accum)
        it = camera_pose_.inverseTransform() * it;

      for (int o = 0; o < _n_objects; o++)
        if (delta_poses_accum.at(o).normT2() <
            parameters_.max_t_update_norm_squared_)
          poses_accum.at(o) = delta_poses_accum.at(o) * poses_accum.at(o);
        else // invalidate
          poses_accum.at(o).setValid(false);

      // transform back to world frame
      for (auto &it : poses_accum)
        it = camera_pose_ * it;

#ifdef TIME_STEPS
      cudaThreadSynchronize();
      cudaEventRecord(end, 0);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, start, end);
      _compTimes.at(12) += (double)elapsed_time;
#endif

#ifdef TIME_STEPS
      cudaEventRecord(start, 0);
#endif

      // render at updated pose (unless last iteration)
      if (it < parameters_.n_icp_outer_it_)
        render(poses_accum);

#ifdef TIME_STEPS
      cudaThreadSynchronize();
      cudaEventRecord(end, 0);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, start, end);
      _compTimes.at(11) += (double)elapsed_time;
#endif
    }

    // update current pose
    _currentPoses = poses_accum;
  }
}

void D_MultipleRigidPoses::updateNormalEquations(
    const util::Device1D<float> &d_flowx, const util::Device1D<float> &d_flowy,
    const util::Device1D<float> &d_ar_flowx,
    const util::Device1D<float> &d_ar_flowy,
    const util::Device2D<float> &d_disparity,
    const std::vector<TranslationRotation3D> &explained_delta_poses,
    std::bitset<32> segments_to_update) {
  // ensure rendering is up-to-date
  render(_currentPoses);

  // store initial Zbuffer as initial model for residual flow
  convertZbufferToZ(d_init_Z_->data(), _multipleRigidModelsOgre->getZBuffer(),
                    _n_cols, _n_rows, _nodal_point_x, _nodal_point_y,
                    parameters_.near_plane_, parameters_.far_plane_);

  // initialize residual flow (will improve later)
  d_res_flowx_->copyFrom(d_flowx);
  d_res_flowy_->copyFrom(d_flowy);
  d_res_ar_flowx_->copyFrom(d_ar_flowx);
  d_res_ar_flowy_->copyFrom(d_ar_flowy);

  // subtract camera-motion-induced flow
  std::vector<float> dT_accum(3 * _n_objects);
  std::vector<float> dR_accum(9 * _n_objects);

  for (int o = 0; o < _n_objects; o++) {
    explained_delta_poses.at(o).getT(&dT_accum[3 * o]);
    explained_delta_poses.at(o).getR_mat(&dR_accum[9 * o]);
  }
  d_delta_T_accum_->copyFrom(dT_accum, 3 * _n_objects);
  d_delta_Rmat_accum_->copyFrom(dR_accum, 9 * _n_objects);

  computeResidualFlow(
      d_res_flowx_->data(), d_res_flowy_->data(), d_res_ar_flowx_->data(),
      d_res_ar_flowy_->data(), d_flowx.data(), d_flowy.data(),
      d_ar_flowx.data(), d_ar_flowy.data(), d_delta_T_accum_->data(),
      d_delta_Rmat_accum_->data(), d_init_Z_->data(),
      _multipleRigidModelsOgre->getSegmentIND(), _n_cols, _n_rows,
      _nodal_point_x, _nodal_point_y, _focal_length_x, _focal_length_y);

  // reset normal equations
  for (auto &it : segment_normal_eqs_)
    it.reset();

  // perform robust iteration
  robustPoseUpdates(d_res_flowx_->data(), d_res_flowy_->data(),
                    d_res_ar_flowx_->data(), d_res_ar_flowy_->data(),
                    d_disparity.data(), d_disparity.pitch(),
                    (int)segments_to_update.to_ulong());

  // prepare normal equations for (external) constrained update
  int min_samples = 1000;
  for (auto &it : _segment_info) {
    // reset if not enough samples or update norm too large
    if (((it.n_values_flow + it.n_values_disparity) < min_samples) &&
        (segment_normal_eqs_.at(it.segment_ind).squaredNormDeltaT() >=
         parameters_.max_t_update_norm_squared_))
      segment_normal_eqs_.at(it.segment_ind).reset();
  }
}

void D_MultipleRigidPoses::setWeights(float w_flow, float w_ar_flow,
                                      float w_disp) {
  parameters_.w_flow_ = w_flow;
  parameters_.w_ar_flow_ = w_ar_flow;
  parameters_.w_disp_ = w_disp;
}

void D_MultipleRigidPoses::getWeights(float &w_flow, float &w_ar_flow,
                                      float &w_disp) const {
  w_flow = parameters_.w_flow_;
  w_ar_flow = parameters_.w_ar_flow_;
  w_disp = parameters_.w_disp_;
}

cudaArray *D_MultipleRigidPoses::getTexture() {
  render(_currentPoses);
  return (_multipleRigidModelsOgre->getTexture());
}

cudaArray *D_MultipleRigidPoses::getSegmentIND() {
  render(_currentPoses);
  return (_multipleRigidModelsOgre->getSegmentIND());
}

cudaArray *D_MultipleRigidPoses::getZbuffer() {
  render(_currentPoses);
  return _multipleRigidModelsOgre->getZBuffer();
}

cudaArray *D_MultipleRigidPoses::getNormalX() {
  render(_currentPoses);
  return _multipleRigidModelsOgre->getNormalX();
}

cudaArray *D_MultipleRigidPoses::getNormalY() {
  render(_currentPoses);
  return _multipleRigidModelsOgre->getNormalY();
}

cudaArray *D_MultipleRigidPoses::getNormalZ() {
  render(_currentPoses);
  return _multipleRigidModelsOgre->getNormalZ();
}

void D_MultipleRigidPoses::setCameraParameters(float focal_length_x,
                                               float focal_length_y,
                                               float nodal_point_x,
                                               float nodal_point_y) {
  render_state_changed_ = true;
  _nodal_point_x = nodal_point_x;
  _nodal_point_y = nodal_point_y;
  _focal_length_x = focal_length_x;
  _focal_length_y = focal_length_y;
  _multipleRigidModelsOgre->updateProjectionMatrix(
      _focal_length_x, _focal_length_y, _nodal_point_x, _nodal_point_y,
      parameters_.near_plane_, parameters_.far_plane_);
}

void
D_MultipleRigidPoses::setCameraPose(const TranslationRotation3D &camera_pose) {
  render_state_changed_ = true;

  // if this is the first camera pose received, make previous and current equal
  previous_camera_pose_ =
      previous_camera_pose_.isValid() ? camera_pose_ : camera_pose;

  camera_pose_ = camera_pose;
  _multipleRigidModelsOgre->updateCameraPose(camera_pose_.rotateX180());
}

bool D_MultipleRigidPoses::isDenseWinner() {

  bool denseWinner = true;
  double thres = parameters_.reliability_threshold_;

  // make sure that sparse does not negatively affect any object's proportion ar
  // valid
  // add some margin to allow minor decrease in reliability (due to noise)
  double margin = parameters_.sparse_intro_allowed_reliability_decrease_;
  bool sparseRelDecrease = false;
  for (int o = 0; o < _n_objects; o++)
    if (_S_ar_flow_prop_valid[o] < (_D_ar_flow_prop_valid[o] - margin))
      sparseRelDecrease = true;

  if (!sparseRelDecrease) {

    // compare reliability at the current sparse object
    int o = _currentSparseObject;

    // we require a pretty high reliability for sparse introduction
    double intro_thres = parameters_.sparse_intro_reliability_threshold_;
    if ((_S_ar_flow_prop_valid[o] > (_D_ar_flow_prop_valid[o] + intro_thres)) ||
        ((_S_ar_flow_prop_valid[o] > intro_thres) &&
         (_D_ar_flow_prop_valid[o] < thres)))
      denseWinner = false;
  }

  return (denseWinner);
}

void D_MultipleRigidPoses::render(
    const std::vector<TranslationRotation3D> &renderPoses) {
  if ((renderPoses != _lastPosesRendered) || (render_state_changed_)) {

    //    util::TimerGPU render_timer;
    _multipleRigidModelsOgre->render(renderPoses);
    //    std::cout << "render time: " << render_timer.read() << " ms\n";
    _lastPosesRendered = renderPoses;
    render_state_changed_ = false;
  }
}

void
D_MultipleRigidPoses::getSegmentLengths(std::vector<int> &lengths,
                                        const std::vector<int> &starting_inds,
                                        int n_segments, int total_length) {

  for (int i = 0; i < (n_segments - 1); i++) {
    int s = starting_inds.at(i);
    if (s >= 0) {

      // find end-point
      int e = starting_inds.at(i + 1);
      for (int j = (i + 2); ((e < 0) && (j < n_segments)); j++)
        e = starting_inds.at(j);

      if (e < 0)
        e = total_length;

      lengths.at(i) = e - s;

    } else

      lengths.at(i) = 0;
  }

  // final segment
  int s = starting_inds.at(n_segments - 1);
  if (s >= 0)
    lengths.at(n_segments - 1) = total_length - s;
  else
    lengths.at(n_segments - 1) = 0;
}

int D_MultipleRigidPoses::getNextSelectedSparseObject(bool dense) {

  std::vector<double> probabilities =
      dense ? _D_ar_flow_prop_valid : _S_ar_flow_prop_valid;

  double sum = 0.0;
  for (int o = 0; o < _n_objects; o++) {
    probabilities.at(o) = 1.0 - probabilities.at(o);
    sum += probabilities.at(o);
  }

  for (int o = 0; o < _n_objects; o++)
    probabilities.at(o) /= sum;

  double r = ((double)rand() / double(RAND_MAX));

  int object = 0;
  double prob_ulim = probabilities.at(object);

  while (r > prob_ulim) {
    object++;
    prob_ulim += probabilities.at(object);
  }

  return (object);
}

void D_MultipleRigidPoses::evaluateARFlowPoseError(
    bool dense, const util::Device1D<float> &d_ar_flowx,
    std::vector<TranslationRotation3D> &poses) {
  if (dense)
    computeARFlowPoseError(d_ar_flowx, poses, _D_ar_flow_prop_valid);
  else
    computeARFlowPoseError(d_ar_flowx, poses, _S_ar_flow_prop_valid);
}

void D_MultipleRigidPoses::computeARFlowPoseError(
    const util::Device1D<float> &d_ar_flowx,
    std::vector<TranslationRotation3D> &poses,
    std::vector<double> &ar_flow_prop_valid) {
  std::vector<double> ar_flow_abs_valid; // discarded
  computeARFlowPoseError(d_ar_flowx, poses, ar_flow_prop_valid,
                         ar_flow_abs_valid);
}

void D_MultipleRigidPoses::computeARFlowPoseError(
    const util::Device1D<float> &d_ar_flowx,
    std::vector<TranslationRotation3D> &poses,
    std::vector<double> &ar_flow_prop_valid,
    std::vector<double> &ar_flow_abs_valid) {
  ar_flow_prop_valid.clear();
  ar_flow_abs_valid.clear();

  render(poses);

  // re-purposing some already allocated data
  auto d_valid_ar_flow_Zbuffer = d_valid_disparity_Zbuffer_.get();
  auto d_valid_Zbuffer = d_valid_disparity_Zbuffer_sub_.get();
  markValidFlowZbufferAndZbufferZeroBased(
      d_valid_ar_flow_Zbuffer->data(), d_valid_Zbuffer->data(),
      d_ar_flowx.data(), _multipleRigidModelsOgre->getSegmentIND(), _n_cols,
      _n_rows, _n_objects);

  // additional buffers required for sorting
  auto d_value = d_ind_disparity_Zbuffer_.get();
  auto d_value_buf = d_ind_disparity_Zbuffer_sub_.get();
  auto d_key_buf = d_extra_disparity_buffer_.get();

  // Radix sort all the indices using the valid marks
  // indices could be ignored here to speed up sorting (but diff is minimal)
  cub_radix_sorter_->sort(*d_valid_ar_flow_Zbuffer, *d_value, *d_key_buf,
                          *d_value_buf);
  pose::extractLabelStartingIndices(d_seg_start_inds_->data(),
                                    d_valid_ar_flow_Zbuffer->data(),
                                    _n_cols * _n_rows, _n_objects);
  std::vector<int> h_seg_start_inds_flow_Zbuffer(_n_objects + 1);
  d_seg_start_inds_->copyTo(h_seg_start_inds_flow_Zbuffer, _n_objects + 1);

  cub_radix_sorter_->sort(*d_valid_Zbuffer, *d_value, *d_key_buf, *d_value_buf);
  pose::extractLabelStartingIndices(d_seg_start_inds_->data(),
                                    d_valid_Zbuffer->data(), _n_cols * _n_rows,
                                    _n_objects);
  std::vector<int> h_seg_start_inds_Zbuffer(_n_objects + 1);
  d_seg_start_inds_->copyTo(h_seg_start_inds_Zbuffer, _n_objects + 1);

  //  for(int o=0;o<(_n_objects+1);o++)
  //    printf("obj %d - flow+z %09d - z
  // %09d\n",o,h_seg_start_inds_flow_Zbuffer[o],h_seg_start_inds_Zbuffer[o]);

  int n_valid_flow_Zbuffer = (h_seg_start_inds_flow_Zbuffer.at(_n_objects) >= 0)
                                 ? h_seg_start_inds_flow_Zbuffer.at(_n_objects)
                                 : (_n_cols * _n_rows);

  _seg_lengths_flow_Zbuffer.assign(_n_objects, 0);

  getSegmentLengths(_seg_lengths_flow_Zbuffer, h_seg_start_inds_flow_Zbuffer,
                    _n_objects, n_valid_flow_Zbuffer);

  int n_valid_Zbuffer = (h_seg_start_inds_Zbuffer.at(_n_objects) >= 0)
                            ? h_seg_start_inds_Zbuffer.at(_n_objects)
                            : (_n_cols * _n_rows);

  _seg_lengths_Zbuffer.assign(_n_objects, 0);
  getSegmentLengths(_seg_lengths_Zbuffer, h_seg_start_inds_Zbuffer, _n_objects,
                    n_valid_Zbuffer);

  //  for(int o=0;o<(_n_objects+1);o++)
  //    printf("obj %d - start+z %09d - length+z %09d - start %09d - length
  // %09d\n",o,h_seg_start_inds_flow_Zbuffer[o],seg_lengths_flow_Zbuffer[o],h_seg_start_inds_Zbuffer[o],seg_lengths_Zbuffer[o]);

  // save proportion valid for each object
  auto bounding_boxes =
      _multipleRigidModelsOgre->getBoundingBoxesInCameraImage(poses);

  for (int o = 0; o < _n_objects; o++) {

    double absolute_count = (double)_seg_lengths_flow_Zbuffer[o];
    double prop = (_seg_lengths_Zbuffer[o] > 0)
                      ? (double)_seg_lengths_flow_Zbuffer[o] /
                            (double)_seg_lengths_Zbuffer[o]
                      : 0.0;

    if (parameters_.check_reliability_) {
      // does the proportion valid AR flow exceed the threshold?
      bool valid = prop > parameters_.reliability_threshold_;

      // is the projected shape 'sufficiently two-dimensional'?
      // the AR flow-based reliability measure fails on near-edge-like object
      // projections
      Eigen::Map<Eigen::Matrix<double, 2, 8> > bb_pixel(
          bounding_boxes.at(o).data());

      // subtract mean
      Eigen::Vector2d mn = bb_pixel.rowwise().mean();
      bb_pixel.colwise() -= mn;

      // covariance
      double ratio = parameters_.max_proportion_projected_bounding_box_;
      Eigen::Matrix2d cov = bb_pixel * bb_pixel.transpose();
      // check validity
      if (std::isfinite(cov(0, 0)) && std::isfinite(cov(0, 1)) &&
          std::isfinite(cov(1, 0)) && std::isfinite(cov(1, 1))) {
        Eigen::EigenSolver<Eigen::Matrix2d> eigen_solver(cov);
        // rotate bounding box
        Eigen::Matrix2d real_eigenvectors = eigen_solver.eigenvectors().real();
        //        Eigen::VectorXd w = eigen_solver.eigenvalues().real();
        bb_pixel = real_eigenvectors.transpose() * bb_pixel;
        // find extent
        auto extent =
            bb_pixel.rowwise().maxCoeff() - bb_pixel.rowwise().minCoeff();
        ratio = extent.maxCoeff() / extent.minCoeff();
      }
      valid =
          valid && (ratio < parameters_.max_proportion_projected_bounding_box_);

      poses.at(o).setValid(valid);
    }

    ar_flow_prop_valid.push_back(prop);
    ar_flow_abs_valid.push_back(absolute_count);
  }
}

void D_MultipleRigidPoses::robustPoseUpdates(
    const float *d_flowx, const float *d_flowy, const float *d_ar_flowx,
    const float *d_ar_flowy, const float *d_disparity, size_t d_disparity_pitch,
    int segments_to_update) {

  cudaArray *d_ZbufferArray = _multipleRigidModelsOgre->getZBuffer();
  cudaArray *d_normalXArray = _multipleRigidModelsOgre->getNormalX();
  cudaArray *d_normalYArray = _multipleRigidModelsOgre->getNormalY();
  cudaArray *d_normalZArray = _multipleRigidModelsOgre->getNormalZ();
  cudaArray *d_segmentINDArray = _multipleRigidModelsOgre->getSegmentIND();

#ifdef TIME_STEPS
  // Setup timers
  cudaEvent_t start, end, start_sub, end_sub;
  float elapsed_time;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventCreate(&start_sub);
  cudaEventCreate(&end_sub);
#endif
#ifdef TIME_STEPS
  cudaEventRecord(start, 0); // preprocessing
#endif

  // initialize normal equations to zero
  std::fill(h_CO_reduced_.begin(), h_CO_reduced_.end(), 0);
  std::fill(h_CD_reduced_.begin(), h_CD_reduced_.end(), 0);
  cudaMemset(d_CO_reduced_->data(), 0,
             _N_CON_FLOW * parameters_.max_objects_ * sizeof(float));
  cudaMemset(d_CD_reduced_->data(), 0,
             _N_CON_DISP * parameters_.max_objects_ * sizeof(float));

  // Determine Zbuffer conversion constants
  // depth = Z_conv1/(Zbuffer+Z_conv2)
  float Z_conv1, Z_conv2;
  get_GL_conv_constants(Z_conv1, Z_conv2, parameters_.far_plane_,
                        parameters_.near_plane_);

// Mark valid locations with segment index
// If a cue's weight equals 0 mark everything invalid
// No initialization required

#ifdef TIME_STEPS
  cudaEventRecord(start_sub, 0);
#endif

  mark_with_zero_based_segmentIND(
      d_valid_flow_Zbuffer_->data(), d_valid_disparity_Zbuffer_->data(),
      d_flowx, d_ar_flowx, (const char *)d_disparity, d_segmentINDArray,
      _n_cols, _n_rows, _n_objects, d_disparity_pitch, parameters_.w_flow_ > 0,
      parameters_.w_ar_flow_ > 0, parameters_.w_disp_ > 0, segments_to_update);

#ifdef TIME_STEPS
  cudaThreadSynchronize();
  cudaEventRecord(end_sub, 0);
  cudaEventSynchronize(end_sub);
  cudaEventElapsedTime(&elapsed_time, start_sub, end_sub);
  _compTimes.at(3) += (double)elapsed_time;
#endif

#ifdef TIME_STEPS
  cudaEventRecord(start_sub, 0);
#endif

  // Radix sort all the indices using the valid marks
  // the _sub_ buffers are only used for double-buffering (ping-pong)
  d_ind_flow_Zbuffer_->copyFrom(*d_linear_ind_);
  d_ind_disparity_Zbuffer_->copyFrom(*d_linear_ind_, _n_cols * _n_rows);

  //  util::TimerGPU sort_timer;
  cub_radix_sorter_->sort(*d_valid_flow_Zbuffer_, *d_ind_flow_Zbuffer_,
                          *d_valid_flow_Zbuffer_sub_, *d_ind_flow_Zbuffer_sub_);
  //  std::cout << "flow sort time: " << sort_timer.read();
  //  sort_timer.reset();
  cub_radix_sorter_->sort(
      *d_valid_disparity_Zbuffer_, *d_ind_disparity_Zbuffer_,
      *d_valid_disparity_Zbuffer_sub_, *d_ind_disparity_Zbuffer_sub_);
//  std::cout << " disp sort time: " << sort_timer.read() << std::endl;

#ifdef TIME_STEPS
  cudaThreadSynchronize();
  cudaEventRecord(end_sub, 0);
  cudaEventSynchronize(end_sub);
  cudaEventElapsedTime(&elapsed_time, start_sub, end_sub);
  _compTimes.at(4) += (double)elapsed_time;
#endif

  // Get starting indices
  pose::extractLabelStartingIndices(d_seg_start_inds_->data(),
                                    d_valid_flow_Zbuffer_->data(),
                                    _n_cols * _n_rows * _N_FLOWS, _n_objects);
  std::vector<int> h_seg_start_inds_flow(_n_objects + 1);
  d_seg_start_inds_->copyTo(h_seg_start_inds_flow, _n_objects + 1);
  pose::extractLabelStartingIndices(d_seg_start_inds_->data(),
                                    d_valid_disparity_Zbuffer_->data(),
                                    _n_cols * _n_rows, _n_objects);
  std::vector<int> h_seg_start_inds_disparity(_n_objects + 1);
  d_seg_start_inds_->copyTo(h_seg_start_inds_disparity, _n_objects + 1);

  //  printf("before sub\n");
  //  for(int i=0;i<(_n_objects+1);i++)
  //    printf("se %d f %+07d d
  // %+07d\n",i,h_seg_start_inds_flow[i],h_seg_start_inds_disparity[i]);

  // Regularly subsample flow and disparity indices if required when a maximum
  // number of samples has been enforced
  // Relative proportion flow/disp will be maintained approximately

  int n_valid_flow_Zbuffer = (h_seg_start_inds_flow.at(_n_objects) >= 0)
                                 ? h_seg_start_inds_flow.at(_n_objects)
                                 : (_n_cols * _n_rows * _N_FLOWS);
  int n_valid_disparity_Zbuffer =
      (h_seg_start_inds_disparity.at(_n_objects) >= 0)
          ? h_seg_start_inds_disparity.at(_n_objects)
          : (_n_cols * _n_rows);

  //  printf("before sub %s\n",cudaGetErrorString(cudaGetLastError()));
  //  printf("val flow %d val disp
  // %d\n",n_valid_flow_Zbuffer,n_valid_disparity_Zbuffer);

  int n_valid_total = n_valid_flow_Zbuffer + n_valid_disparity_Zbuffer;

  if (n_valid_total > parameters_.max_samples_) {

    // proportionally subsample flow and disparity
    double sub_factor =
        (double)parameters_.max_samples_ / (double)n_valid_total;
    double inv_sub_factor = 1.0 / sub_factor;

    // flow
    int n_valid_flow_Zbuffer_sub =
        (int)floor((double)n_valid_flow_Zbuffer * sub_factor);

    if (n_valid_flow_Zbuffer_sub > 0) {
      subsample_ind_and_labels(
          d_ind_flow_Zbuffer_sub_->data(), d_ind_flow_Zbuffer_->data(),
          d_valid_flow_Zbuffer_sub_->data(), d_valid_flow_Zbuffer_->data(),
          n_valid_flow_Zbuffer_sub, (float)inv_sub_factor);

      // update all regular variables to account for subsampling
      // swap full and subsampled storage!
      n_valid_flow_Zbuffer = n_valid_flow_Zbuffer_sub;
      d_ind_flow_Zbuffer_->swap(*d_ind_flow_Zbuffer_sub_);
      d_valid_flow_Zbuffer_->swap(*d_valid_flow_Zbuffer_sub_);

      // update segment starting indices
      pose::extractLabelStartingIndices(d_seg_start_inds_->data(),
                                        d_valid_flow_Zbuffer_->data(),
                                        n_valid_flow_Zbuffer, _n_objects);
      d_seg_start_inds_->copyTo(h_seg_start_inds_flow, _n_objects + 1);
      //      printf("valsub flow %d\n",n_valid_flow_Zbuffer);
    }

    // disparity
    int n_valid_disparity_Zbuffer_sub =
        (int)floor((double)n_valid_disparity_Zbuffer * sub_factor);

    if (n_valid_disparity_Zbuffer_sub > 0) {
      subsample_ind_and_labels(d_ind_disparity_Zbuffer_sub_->data(),
                               d_ind_disparity_Zbuffer_->data(),
                               d_valid_disparity_Zbuffer_sub_->data(),
                               d_valid_disparity_Zbuffer_->data(),
                               n_valid_disparity_Zbuffer_sub,
                               (float)inv_sub_factor);

      // update all regular variables to account for subsampling
      // swap full and subsampled storage!
      n_valid_disparity_Zbuffer = n_valid_disparity_Zbuffer_sub;
      d_ind_disparity_Zbuffer_->swap(*d_ind_disparity_Zbuffer_sub_);
      d_valid_disparity_Zbuffer_->swap(*d_valid_disparity_Zbuffer_sub_);

      // update segment starting indices
      pose::extractLabelStartingIndices(d_seg_start_inds_->data(),
                                        d_valid_disparity_Zbuffer_->data(),
                                        n_valid_disparity_Zbuffer, _n_objects);
      d_seg_start_inds_->copyTo(h_seg_start_inds_disparity, _n_objects + 1);
      //      printf("valsub disp %d\n",n_valid_disparity_Zbuffer);
    }

    //    printf("after sub\n");
    //    for(int i=0;i<(_n_objects+1);i++)
    //      printf("se %d f %+07d d
    // %+07d\n",i,h_seg_start_inds_flow[i],h_seg_start_inds_disparity[i]);

  } // subsample

  // compute segment lengths
  std::vector<int> seg_lengths_flow(_n_objects);
  getSegmentLengths(seg_lengths_flow, h_seg_start_inds_flow, _n_objects,
                    n_valid_flow_Zbuffer);
  std::vector<int> seg_lengths_disparity(_n_objects);
  getSegmentLengths(seg_lengths_disparity, h_seg_start_inds_disparity,
                    _n_objects, n_valid_disparity_Zbuffer);

//  printf("after subsampling\n");
//  printf("n_f %07d n_d %07d\n", n_valid_flow_Zbuffer,
//         n_valid_disparity_Zbuffer);
//  for (int i = 0; i < _n_objects; i++)
//    printf("se %d f %+07d (%07d) d %+07d (%07d)\n", i,
// h_seg_start_inds_flow[i],
//           seg_lengths_flow[i], h_seg_start_inds_disparity[i],
//           seg_lengths_disparity[i]);

#ifdef TIME_STEPS
  cudaEventRecord(start_sub, 0);
#endif

  // Gather flow and disparity
  if (n_valid_flow_Zbuffer > 0)
    gather_valid_flow_Zbuffer(
        d_flow_compact_->data(), d_Zbuffer_flow_compact_->data(), d_flowx,
        d_flowy, d_ar_flowx, d_ar_flowy, d_ind_flow_Zbuffer_->data(),
        d_ZbufferArray, n_valid_flow_Zbuffer, _n_cols, _n_rows, Z_conv1,
        Z_conv2);

  if (n_valid_disparity_Zbuffer > 0)
    gather_valid_disparity_Zbuffer(
        d_disparity_compact_->data(), d_Zbuffer_normals_compact_->data(),
        (const char *)d_disparity, d_ind_disparity_Zbuffer_->data(),
        d_ZbufferArray, d_normalXArray, d_normalYArray, d_normalZArray,
        n_valid_disparity_Zbuffer, _n_cols, _n_rows, Z_conv1, Z_conv2,
        d_disparity_pitch);

#ifdef TIME_STEPS
  cudaThreadSynchronize();
  cudaEventRecord(end_sub, 0);
  cudaEventSynchronize(end_sub);
  cudaEventElapsedTime(&elapsed_time, start_sub, end_sub);
  _compTimes.at(5) += (double)elapsed_time;
#endif

  // Gather index information
  _segment_info.clear();

  for (int o = 0; o < _n_objects; o++) {
    int nf = seg_lengths_flow[o];
    int nd = seg_lengths_disparity[o];
    if ((nf > 0) || (nd > 0)) {
      SegmentINFO tmp;
      tmp.segment_ind = o;
      tmp.n_values_flow = nf;
      tmp.n_values_disparity = nd;
      tmp.start_ind_flow = h_seg_start_inds_flow[o];
      tmp.start_ind_disparity = h_seg_start_inds_disparity[o];
      _segment_info.push_back(tmp);
    }
  }

//    printf("------------------------------------\n");
//    for(int o=0;o<_segment_info.size();o++) {
//      SegmentINFO &tmp = _segment_info.at(o);
//      printf("seg %d f_start %06d f_size %06d d_start %06d d_size
// %06d\n",tmp.segment_ind,tmp.start_ind_flow,tmp.n_values_flow,tmp.start_ind_disparity,tmp.n_values_disparity);
//    }
//    printf("------------------------------------\n");

#ifdef TIME_STEPS
  cudaThreadSynchronize();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, start, end);
  //  printf("preproc time : %2.4f ms\n",elapsed_time);
  _compTimes.at(0) += (double)elapsed_time;
#endif

  // preparatory book-keeping

  int n_segments = _segment_info.size();

  if (n_segments > 0) {

    std::vector<int> n_values_flow(n_segments);
    std::vector<int> start_ind_flow(n_segments);
    std::vector<int> n_values_disparity(n_segments);
    std::vector<int> start_ind_disparity(n_segments);
    std::vector<int> start_ind_res_flow(n_segments);
    std::vector<int> start_ind_res_disparity(n_segments);

    for (int s = 0; s < n_segments; s++) {
      SegmentINFO &tmp = _segment_info.at(s);
      n_values_flow.at(s) = tmp.n_values_flow;
      start_ind_flow.at(s) = tmp.start_ind_flow;
      n_values_disparity.at(s) = tmp.n_values_disparity;
      start_ind_disparity.at(s) = tmp.start_ind_disparity;

      start_ind_res_flow.at(s) = (s > 0) ? (start_ind_res_disparity.at(s - 1) +
                                            n_values_disparity.at(s - 1))
                                         : 0;
      start_ind_res_disparity.at(s) =
          start_ind_res_flow.at(s) + n_values_flow.at(s);
    }

    // Compute offset between residual input and output indices. Residuals are
    // stored as [seg1_flow seg1_disp seg2_flow seg2_disp ... ] to facilitate
    // median extraction

    std::vector<int> offset_ind_res_flow(n_segments);
    std::vector<int> offset_ind_res_disparity(n_segments);

    for (int s = 0; s < n_segments; s++) {
      offset_ind_res_flow.at(s) =
          start_ind_res_flow.at(s) - start_ind_flow.at(s);
      offset_ind_res_disparity.at(s) =
          start_ind_res_disparity.at(s) - start_ind_disparity.at(s);
    }

    // Create segment translation table (map zero-based original segment indices
    // to compressed segment indices)

    std::vector<int> segment_translation_table(_n_objects);
    for (int s = 0; s < n_segments; s++)
      segment_translation_table.at(_segment_info.at(s).segment_ind) = s;

    d_segment_translation_table_->copyFrom(segment_translation_table,
                                           _n_objects);
    d_n_values_flow_->copyFrom(n_values_flow, n_segments);
    d_start_ind_flow_->copyFrom(start_ind_flow, n_segments);
    d_n_values_disparity_->copyFrom(n_values_disparity, n_segments);
    d_start_ind_disparity_->copyFrom(start_ind_disparity, n_segments);
    d_offset_ind_res_flow_->copyFrom(offset_ind_res_flow, n_segments);
    d_offset_ind_res_disparity_->copyFrom(offset_ind_res_disparity, n_segments);
  }

  // take care here since _d_CO and _d_CD's maximum sizes are not exactly
  // determined in this way (especially the times 4)

  dim3 threadBlock_normal(64, 1, 1);
  int gridDim_x_normal_equations =
      (n_segments > 0)
          ? divUp(_MAX_N_VAL_ACCUM, n_segments * threadBlock_normal.x) * 4
          : 0;
  dim3 blockGrid_normal(gridDim_x_normal_equations, n_segments);
  //  dim3 threadBlock_reduce_64(64,1);
  //  dim3 blockGrid_reduce_flow(_N_CON_FLOW, n_segments);
  //  dim3 blockGrid_reduce_disparity(_N_CON_DISP, n_segments);

  dim3 threadBlock_reduce_64_mult(64, 4);
  dim3 blockGrid_reduce_flow_mult(divUp(_N_CON_FLOW, 4), n_segments);
  dim3 blockGrid_reduce_disparity_mult(divUp(_N_CON_DISP, 4), n_segments);

/****************/
/* OLS Estimate */
/****************/

#ifdef TIME_STEPS
  cudaEventRecord(start, 0); // ols-time
#endif

  if (n_segments > 0) {

#ifdef TIME_STEPS
    cudaEventRecord(start_sub, 0);
#endif

    normal_eqs_flow(blockGrid_normal, threadBlock_normal, d_CO_->data(),
                    d_flow_compact_->data(), d_Zbuffer_flow_compact_->data(),
                    d_ind_flow_Zbuffer_->data(), _focal_length_x,
                    _focal_length_y, _nodal_point_x, _nodal_point_y, _n_rows,
                    _n_cols, d_n_values_flow_->data(),
                    d_start_ind_flow_->data());
    normal_eqs_disparity(blockGrid_normal, threadBlock_normal, d_CD_->data(),
                         d_disparity_compact_->data(),
                         d_Zbuffer_normals_compact_->data(),
                         d_ind_disparity_Zbuffer_->data(), _focal_length_x,
                         _focal_length_y, _nodal_point_x, _nodal_point_y,
                         _baseline, _n_cols, d_n_values_disparity_->data(),
                         d_start_ind_disparity_->data(), parameters_.w_disp_);

#ifdef TIME_STEPS
    cudaThreadSynchronize();
    cudaEventRecord(end_sub, 0);
    cudaEventSynchronize(end_sub);
    cudaEventElapsedTime(&elapsed_time, start_sub, end_sub);
    _compTimes.at(6) += (double)elapsed_time;
#endif

#ifdef TIME_STEPS
    cudaEventRecord(start_sub, 0);
#endif

    reduce_normal_eqs_64_mult_constr(blockGrid_reduce_flow_mult,
                                     threadBlock_reduce_64_mult,
                                     d_CO_reduced_->data(), d_CO_->data(),
                                     gridDim_x_normal_equations, _N_CON_FLOW);
    reduce_normal_eqs_64_mult_constr(blockGrid_reduce_disparity_mult,
                                     threadBlock_reduce_64_mult,
                                     d_CD_reduced_->data(), d_CD_->data(),
                                     gridDim_x_normal_equations, _N_CON_DISP);

//    reduce_normal_eqs_64_GPU<<<blockGrid_reduce_flow,threadBlock_reduce_64>>>(_d_CO_reduced,
// _d_CO, gridDim_x_normal_equations);
//    reduce_normal_eqs_64_GPU<<<blockGrid_reduce_disparity,threadBlock_reduce_64>>>(_d_CD_reduced,
// _d_CD, gridDim_x_normal_equations);

#ifdef TIME_STEPS
    cudaThreadSynchronize();
    cudaEventRecord(end_sub, 0);
    cudaEventSynchronize(end_sub);
    cudaEventElapsedTime(&elapsed_time, start_sub, end_sub);
    _compTimes.at(7) += (double)elapsed_time;
#endif
  }

  d_CO_reduced_->copyTo(h_CO_reduced_);
  d_CD_reduced_->copyTo(h_CD_reduced_);

#ifdef TIME_STEPS
  cudaEventRecord(start_sub, 0);
#endif

  // Solve systems
  std::vector<float> dTdR(6 * n_segments);
  for (int s = 0; s < n_segments; s++) {
    int curr_part = _segment_info.at(s).segment_ind;
    segment_normal_eqs_.at(curr_part).compose(
        &h_CO_reduced_.at(_N_CON_FLOW * s), &h_CD_reduced_.at(_N_CON_DISP * s));
    segment_normal_eqs_.at(curr_part).solve(&dTdR.at(6 * s));
  }

#ifdef TIME_STEPS
  cudaThreadSynchronize();
  cudaEventRecord(end_sub, 0);
  cudaEventSynchronize(end_sub);
  cudaEventElapsedTime(&elapsed_time, start_sub, end_sub);
  _compTimes.at(8) += (double)elapsed_time;
#endif

  // Copy to device memory
  d_dTR_->copyFrom(dTdR, 6 * n_segments);

  //  for(int s=0;s<n_segments;s++) {
  //    printf("seg %d - ",_segment_info.at(s).segment_ind);
  //    for(int i=0;i<6;i++)
  //      printf("%+03.4f ",dTdR[s*6+i]);
  //    printf("\n");
  //  }

  // Store OLS estimates
  _OLSDeltaPoses.clear();
  for (int i = 0; i < n_segments; i++)
    _OLSDeltaPoses.push_back(TranslationRotation3D(&dTdR[i * 6]));

#ifdef TIME_STEPS
  cudaThreadSynchronize();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, start, end);
  //  printf("OLS time : %2.4f ms\n",elapsed_time);
  _compTimes.at(1) += (double)elapsed_time;
#endif

/****************/
/* M-estimation */
/****************/

#ifdef TIME_STEPS
  cudaEventRecord(start, 0); // robust-time
#endif

  for (int it = 0; it < parameters_.n_icp_inner_it_; it++) {

    // Compute flow and disparity residuals

    if (n_segments > 0) {

#ifdef TIME_STEPS
      cudaEventRecord(start_sub, 0);
#endif

      dim3 threadBlock_res(256, 1);

      if (n_valid_flow_Zbuffer > 0) {
        dim3 blockGrid_res(divUp(n_valid_flow_Zbuffer, threadBlock_res.x), 1);
        flow_absolute_residual_scalable(
            blockGrid_res, threadBlock_res, d_abs_res_->data(),
            d_flow_compact_->data(), d_Zbuffer_flow_compact_->data(),
            d_ind_flow_Zbuffer_->data(), d_valid_flow_Zbuffer_->data(),
            _focal_length_x, _focal_length_y, _nodal_point_x, _nodal_point_y,
            _n_rows, _n_cols, n_valid_flow_Zbuffer,
            d_offset_ind_res_flow_->data(),
            d_segment_translation_table_->data(), parameters_.w_flow_,
            parameters_.w_ar_flow_, d_dTR_->data());
      }

      if (n_valid_disparity_Zbuffer > 0) {
        dim3 blockGrid_res(divUp(n_valid_disparity_Zbuffer, threadBlock_res.x),
                           1);
        disp_absolute_residual_scalable(
            blockGrid_res, threadBlock_res, d_abs_res_->data(),
            d_disparity_compact_->data(), d_Zbuffer_normals_compact_->data(),
            d_ind_disparity_Zbuffer_->data(),
            d_valid_disparity_Zbuffer_->data(), _focal_length_x,
            _focal_length_y, _nodal_point_x, _nodal_point_y, _baseline, _n_cols,
            n_valid_disparity_Zbuffer, d_offset_ind_res_disparity_->data(),
            d_segment_translation_table_->data(), parameters_.w_disp_,
            d_dTR_->data());
      }

#ifdef TIME_STEPS
      cudaThreadSynchronize();
      cudaEventRecord(end_sub, 0);
      cudaEventSynchronize(end_sub);
      cudaEventElapsedTime(&elapsed_time, start_sub, end_sub);
      _compTimes.at(9) += (double)elapsed_time;
#endif

#ifdef TIME_STEPS
      cudaEventRecord(start_sub, 0);
#endif

      // Median absolute residuals
      std::vector<float> abs_res_scales(n_segments);
      int pp[n_segments];
      for (int s = 0; s < n_segments; s++) {
        SegmentINFO &tmp = _segment_info.at(s);
        pp[s] = tmp.n_values_flow + tmp.n_values_disparity;
      }
      approx_multiple_medians_shuffle_cuda(
          abs_res_scales.data(), d_abs_res_->data(), d_median_tmp_->data(),
          d_random_numbers_->data(), pp, n_segments, d_median_n_in_->data(),
          d_median_start_inds_->data());

      for (int s = 0; s < n_segments; s++)
        abs_res_scales.at(s) *= 6.9460f;

      d_abs_res_scales_->copyFrom(abs_res_scales, n_segments);

#ifdef TIME_STEPS
      cudaThreadSynchronize();
      cudaEventRecord(end_sub, 0);
      cudaEventSynchronize(end_sub);
      cudaEventElapsedTime(&elapsed_time, start_sub, end_sub);
      _compTimes.at(10) += (double)elapsed_time;
#endif

//  for(int s=0;s<n_segments;s++)
//    printf("%2.5f ",abs_res_scales[s]);
//  printf("\n");

// Weighted constraints

#ifdef TIME_STEPS
      cudaEventRecord(start_sub, 0);
#endif

      normal_eqs_flow_weighted(
          blockGrid_normal, threadBlock_normal, d_CO_->data(),
          d_flow_compact_->data(), d_Zbuffer_flow_compact_->data(),
          d_ind_flow_Zbuffer_->data(), _focal_length_x, _focal_length_y,
          _nodal_point_x, _nodal_point_y, _n_rows, _n_cols,
          d_n_values_flow_->data(), d_start_ind_flow_->data(),
          d_abs_res_scales_->data(), parameters_.w_flow_,
          parameters_.w_ar_flow_, d_dTR_->data());

      normal_eqs_disparity_weighted(
          blockGrid_normal, threadBlock_normal, d_CD_->data(),
          d_disparity_compact_->data(), d_Zbuffer_normals_compact_->data(),
          d_ind_disparity_Zbuffer_->data(), _focal_length_x, _focal_length_y,
          _nodal_point_x, _nodal_point_y, _baseline, _n_cols,
          d_n_values_disparity_->data(), d_start_ind_disparity_->data(),
          d_abs_res_scales_->data(), parameters_.w_disp_, d_dTR_->data());

#ifdef TIME_STEPS
      cudaThreadSynchronize();
      cudaEventRecord(end_sub, 0);
      cudaEventSynchronize(end_sub);
      cudaEventElapsedTime(&elapsed_time, start_sub, end_sub);
      _compTimes.at(6) += (double)elapsed_time;
#endif

#ifdef TIME_STEPS
      cudaEventRecord(start_sub, 0);
#endif

      reduce_normal_eqs_64_mult_constr(blockGrid_reduce_flow_mult,
                                       threadBlock_reduce_64_mult,
                                       d_CO_reduced_->data(), d_CO_->data(),
                                       gridDim_x_normal_equations, _N_CON_FLOW);
      reduce_normal_eqs_64_mult_constr(blockGrid_reduce_disparity_mult,
                                       threadBlock_reduce_64_mult,
                                       d_CD_reduced_->data(), d_CD_->data(),
                                       gridDim_x_normal_equations, _N_CON_DISP);

//      reduce_normal_eqs_64_GPU<<<blockGrid_reduce_flow,threadBlock_reduce_64>>>(_d_CO_reduced,
// _d_CO, gridDim_x_normal_equations);
//      reduce_normal_eqs_64_GPU<<<blockGrid_reduce_disparity,threadBlock_reduce_64>>>(_d_CD_reduced,
// _d_CD, gridDim_x_normal_equations);

#ifdef TIME_STEPS
      cudaThreadSynchronize();
      cudaEventRecord(end_sub, 0);
      cudaEventSynchronize(end_sub);
      cudaEventElapsedTime(&elapsed_time, start_sub, end_sub);
      _compTimes.at(7) += (double)elapsed_time;
#endif
    }

    d_CO_reduced_->copyTo(h_CO_reduced_);
    d_CD_reduced_->copyTo(h_CD_reduced_);

#ifdef TIME_STEPS
    cudaEventRecord(start_sub, 0);
#endif

    // Solve systems
    for (int s = 0; s < n_segments; s++) {
      int curr_part = _segment_info.at(s).segment_ind;
      segment_normal_eqs_.at(curr_part)
          .compose(&h_CO_reduced_.at(_N_CON_FLOW * s),
                   &h_CD_reduced_.at(_N_CON_DISP * s));
      segment_normal_eqs_.at(curr_part).solve(&dTdR.at(6 * s));
    }

#ifdef TIME_STEPS
    cudaThreadSynchronize();
    cudaEventRecord(end_sub, 0);
    cudaEventSynchronize(end_sub);
    cudaEventElapsedTime(&elapsed_time, start_sub, end_sub);
    _compTimes.at(8) += (double)elapsed_time;
#endif

    // Copy to device memory
    d_dTR_->copyFrom(dTdR, 6 * n_segments);

    //    printf("ROBUST ITERATION %d\n",it);
    //    printf("-------------------\n");
    //    for(int s=0;s<n_segments;s++) {
    //      printf("seg %d - ",_segment_info.at(s).segment_ind);
    //      for(int i=0;i<6;i++)
    //        printf("%+03.4f ",dTdR[s*6+i]);
    //      printf("\n");
    //    }
  }

  // Store robust estimates
  _robustDeltaPoses.clear();
  for (int i = 0; i < n_segments; i++)
    _robustDeltaPoses.push_back(TranslationRotation3D(&dTdR[i * 6]));

#ifdef TIME_STEPS
  cudaThreadSynchronize();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, start, end);
  //  printf("robust time : %2.4f ms\n",elapsed_time);
  _compTimes.at(2) += (double)elapsed_time;
#endif
}
}
