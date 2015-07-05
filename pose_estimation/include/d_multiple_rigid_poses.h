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

#pragma once

//#define TIME_STEPS
#include <memory>
#include <bitset>
#include <cuda_runtime.h>
#include <translation_rotation_3d.h>
#include <multiple_rigid_models_ogre.h>
#include <cub_radix_sorter.h>
//#include <CudaUtilities.h>
#include <device_1d.h>
#include <device_2d.h>
#include <normal_equations.h>

namespace pose {

/* ! \brief Multiple Rigid Object Poses Estimation

  Iteratively updates multiple rigid model poses to match dense optical flow and
  stereo estimates.
  Operates as an engine and can be queried multiple times.
  Is connected to the rendering object and to a flow/stereo(prior) object.
  It controls the rendering object, and can setup the prior of the stereo object

*/

class D_MultipleRigidPoses {

public:
  struct Parameters {
    Parameters()
        : n_icp_outer_it_{ 3 }, n_icp_inner_it_{ 3 }, w_flow_{ 1.0f },
          w_ar_flow_{ 1.0f }, w_disp_{ 1.0f }, max_samples_{ 500000 },
          near_plane_{ .001f }, far_plane_{ 10.0f },
          reliability_threshold_{ 0.15 },
          max_proportion_projected_bounding_box_{ 10.0 },
          sparse_intro_reliability_threshold_{ 0.30 },
          sparse_intro_allowed_reliability_decrease_{ 0.02 },
          max_t_update_norm_squared_{ 0.2 * 0.2 }, check_reliability_{ true } {
      setKeyBits(4);
    }
    void setKeyBits(int key_bits) {
      key_bits_ = key_bits;
      max_objects_ = (int)pow(2.0, (double)key_bits_);
    }

    int getKeyBits() const { return key_bits_; }

    int n_icp_outer_it_;
    int n_icp_inner_it_;
    float w_flow_;
    float w_ar_flow_;
    float w_disp_;
    int max_samples_;
    int max_objects_;
    float near_plane_;
    float far_plane_;
    double reliability_threshold_;
    double max_proportion_projected_bounding_box_;
    double sparse_intro_reliability_threshold_;
    double sparse_intro_allowed_reliability_decrease_;
    double max_t_update_norm_squared_;
    bool check_reliability_;

  private:
    int key_bits_;
  };

  D_MultipleRigidPoses(int n_cols, int n_rows, float nodal_point_x,
                       float nodal_point_y, float focal_length_x,
                       float focal_length_y, float baseline,
                       Parameters parameters = Parameters());

  void addModel(const char *obj_filename, float obj_scale,
                TranslationRotation3D initial_pose);

  // removes all models from pose tracker
  void removeAllModels();

  /*!
   * \brief update poses based on low level cues
   * \param d_flowx
   * \param d_flowy
   * \param d_ar_flowx
   * \param d_ar_flowy
   * \param d_disparity
   * \param segments_to_update which segments to update (zero-based and defaults
   * to everything)
   */
  void update(const util::Device1D<float> &d_flowx,
              const util::Device1D<float> &d_flowy,
              const util::Device1D<float> &d_ar_flowx,
              const util::Device1D<float> &d_ar_flowy,
              const util::Device2D<float> &d_disparity,
              std::bitset<32> segments_to_update = 4294967295);

  /*!
   * \brief perform a single external iteration so that the normal equations are
   * prepared for a subsequent
   * constrained update (performed externally)
   * \param d_flowx
   * \param d_flowy
   * \param d_ar_flowx
   * \param d_ar_flowy
   * \param d_disparity
   * \param explained_delta_poses: each object's pose change that is already
   * explained
   *          (e.g. due to camera motion and/or estimates from previous
   * iterations)
   *          it will be subtracted from the flow before computing the normal
   * equations
   * \param segments_to_update
   */
  void updateNormalEquations(
      const util::Device1D<float> &d_flowx,
      const util::Device1D<float> &d_flowy,
      const util::Device1D<float> &d_ar_flowx,
      const util::Device1D<float> &d_ar_flowy,
      const util::Device2D<float> &d_disparity,
      const std::vector<pose::TranslationRotation3D> &explained_delta_poses,
      std::bitset<32> segments_to_update = 4294967295);

  /*! \brief Computes prop AR flow for each segment and invalidates poses with
   * AR flowprop < _reliabilityThreshold if _checkReliability = true
  */
  void evaluateARFlowPoseError(bool dense,
                               const util::Device1D<float> &d_ar_flowx,
                               std::vector<TranslationRotation3D> &poses);

  /*!
   * \brief computeARFlowPoseError: evaluation function useful for external
   * reliability management
   * \param d_ar_flowx
   * \param poses: may be invalidated if _checkReliability is set
   * \param ar_flow_prop_valid: the proportion valid AR flow
   */
  void computeARFlowPoseError(const util::Device1D<float> &d_ar_flowx,
                              std::vector<TranslationRotation3D> &poses,
                              std::vector<double> &ar_flow_prop_valid);

  bool isDenseWinner();

  std::vector<TranslationRotation3D> getPoses() const {
    return (_currentPoses);
  }

  std::vector<std::vector<double> > getBoundingBoxesInCameraFrame() {
    return _multipleRigidModelsOgre->getBoundingBoxesInCameraFrame(
        _currentPoses);
  }

  std::vector<std::vector<double> > getBoundingBoxesInCameraImage() {
    return _multipleRigidModelsOgre->getBoundingBoxesInCameraImage(
        _currentPoses);
  }

  TranslationRotation3D getSparsePose() const { return (_currentSparsePose); }
  int getSparsePoseObject() const { return (_currentSparseObject); }

  /* ! \brief Determine  sparse object to be estimated next (probabilistically
   * based on current reliability (dense or sparse)
*/
  int getNextSelectedSparseObject(bool dense);

  Ogre::SceneManager *getSceneManager() {
    return _multipleRigidModelsOgre->getSceneManager();
  }

  cudaArray *getTexture();
  cudaArray *getSegmentIND();
  cudaArray *getZbuffer();
  cudaArray *getNormalX();
  cudaArray *getNormalY();
  cudaArray *getNormalZ();

  const Parameters &getParameters() const { return parameters_; }

  int getNObjects() const { return (_n_objects); }
  int getNOuterIt() const { return (parameters_.n_icp_outer_it_); }
  int getNInnerIt() const { return (parameters_.n_icp_inner_it_); }
  void getWeights(float &w_flow, float &w_ar_flow, float &w_disp) const;
  int getMaxSamples() const { return (parameters_.max_samples_); }
  int getWidth() const { return (_n_cols); }
  int getHeight() const { return (_n_rows); }
  //  float getFocalLength() const { return(_focal_length); }
  float getBaseline() const { return (_baseline); }

  std::vector<pose::NormalEquations> getNormalEquations() const {
    return segment_normal_eqs_;
  }

  bool getCheckReliability() const { return (parameters_.check_reliability_); }

  void setPoses(std::vector<TranslationRotation3D> &newPoses) {
    _currentPoses = newPoses;
  }
  void setSparsePose(TranslationRotation3D &newSparsePose, int sparseObject) {
    _currentSparsePose = newSparsePose;
    _currentSparseObject = sparseObject;
  }

  void setCheckReliability(bool checkReliability) {
    parameters_.check_reliability_ = checkReliability;
  }
  void setCameraParameters(float focal_length_x, float focal_length_y,
                           float nodal_point_x, float nodal_point_y);
  void setCameraPose(const TranslationRotation3D &camera_pose);
  //  void setFocalLength(float focal_length) { _focal_length = focal_length; }
  void setBaseline(float baseline) { _baseline = baseline; }
  void setNOuterIt(int nOuterIt) { parameters_.n_icp_outer_it_ = nOuterIt; }
  void setNInnerIt(int nInnerIt) { parameters_.n_icp_inner_it_ = nInnerIt; }
  void setWeights(float w_flow, float w_ar_flow, float w_disp);
  void setMaxSamples(int maxSamples) { parameters_.max_samples_ = maxSamples; }

  void setRenderStateChanged(bool render_state_changed) {
    render_state_changed_ = render_state_changed;
  }

  void enable() { _running = true; }
  void disable() { _running = false; }

private:
  void robustPoseUpdates(const float *d_flowx, const float *d_flowy,
                         const float *d_ar_flowx, const float *d_ar_flowy,
                         const float *d_disparity, size_t d_disparity_pitch,
                         int segments_to_update);

  void render(const std::vector<TranslationRotation3D> &renderPoses);

  void getSegmentLengths(std::vector<int> &lengths,
                         const std::vector<int> &starting_inds, int n_segments,
                         int total_length);

  Parameters parameters_;

  struct SegmentINFO {
    int segment_ind;
    int start_ind_flow;
    int start_ind_disparity;
    int n_values_flow;
    int n_values_disparity;
  };

  std::vector<SegmentINFO> _segment_info;
  std::vector<TranslationRotation3D> _OLSDeltaPoses, _robustDeltaPoses;
  std::vector<pose::NormalEquations> segment_normal_eqs_;

  static const int _N_CON_FLOW =
      23; // number of unique values in flow normal equations
  static const int _N_CON_DISP =
      27; // number of unique values in disparity normal equations
  static const int _MAX_N_VAL_ACCUM =
      1280; // number of samples after accumulation
  static const int _N_FLOWS =
      2; // number of flow inputs (real image and augmented reality image)

  const int _n_cols;
  const int _n_rows;
  float _nodal_point_x;
  float _nodal_point_y;
  float _focal_length_x, _focal_length_y;
  float _baseline;

  const std::unique_ptr<util::CubRadixSorter<unsigned int, int> >
  cub_radix_sorter_;

  const std::unique_ptr<MultipleRigidModelsOgre> _multipleRigidModelsOgre;

  bool render_state_changed_;
  pose::TranslationRotation3D camera_pose_, previous_camera_pose_;
  std::vector<TranslationRotation3D> _currentPoses;
  std::vector<TranslationRotation3D> _lastPosesRendered;
  TranslationRotation3D _currentSparsePose; // only one sparse pose is stored
  int _currentSparseObject;
  int _n_objects;
  std::vector<double> _D_ar_flow_prop_valid, _S_ar_flow_prop_valid;
  std::vector<int> _seg_lengths_Zbuffer, _seg_lengths_flow_Zbuffer;

  std::vector<double> _compTimes;

  util::Device1D<int>::Ptr d_linear_ind_;
  util::Device1D<unsigned int>::Ptr d_valid_flow_Zbuffer_;
  util::Device1D<unsigned int>::Ptr d_valid_disparity_Zbuffer_;
  util::Device1D<unsigned int>::Ptr d_valid_flow_Zbuffer_sub_;
  util::Device1D<unsigned int>::Ptr d_valid_disparity_Zbuffer_sub_;
  util::Device1D<unsigned int>::Ptr d_extra_disparity_buffer_;

  util::Device1D<int>::Ptr d_ind_flow_Zbuffer_;
  util::Device1D<int>::Ptr d_ind_disparity_Zbuffer_;
  util::Device1D<int>::Ptr d_ind_flow_Zbuffer_sub_;
  util::Device1D<int>::Ptr d_ind_disparity_Zbuffer_sub_;

  // starting indices sorted segments
  util::Device1D<int>::Ptr d_seg_start_inds_;

  util::Device1D<float2>::Ptr d_flow_compact_;
  util::Device1D<float>::Ptr d_Zbuffer_flow_compact_;
  util::Device1D<float>::Ptr d_disparity_compact_;
  util::Device1D<float4>::Ptr d_Zbuffer_normals_compact_;
  util::Device1D<int>::Ptr d_n_values_flow_;
  util::Device1D<int>::Ptr d_start_ind_flow_;
  util::Device1D<int>::Ptr d_n_values_disparity_;
  util::Device1D<int>::Ptr d_start_ind_disparity_;

  util::Device1D<float>::Ptr d_CO_;
  util::Device1D<float>::Ptr d_CO_reduced_;
  util::Device1D<float>::Ptr d_CD_;
  util::Device1D<float>::Ptr d_CD_reduced_;
  // store flow+disp residuals together
  util::Device1D<float>::Ptr d_abs_res_;

  std::vector<float> h_CO_reduced_;
  std::vector<float> h_CD_reduced_;

  util::Device1D<int>::Ptr d_offset_ind_res_flow_;
  util::Device1D<int>::Ptr d_offset_ind_res_disparity_;
  util::Device1D<float>::Ptr d_dTR_;
  util::Device1D<float>::Ptr d_delta_T_accum_;
  util::Device1D<float>::Ptr d_delta_Rmat_accum_;
  util::Device1D<int>::Ptr d_segment_translation_table_;

  util::Device1D<float>::Ptr d_random_numbers_;
  util::Device1D<float>::Ptr d_median_tmp_;
  util::Device1D<float>::Ptr d_abs_res_scales_;
  util::Device1D<int>::Ptr d_median_n_in_;
  util::Device1D<int>::Ptr d_median_start_inds_;

  util::Device1D<float>::Ptr d_init_Z_;
  util::Device1D<float>::Ptr d_res_flowx_;
  util::Device1D<float>::Ptr d_res_flowy_;
  util::Device1D<float>::Ptr d_res_ar_flowx_;
  util::Device1D<float>::Ptr d_res_ar_flowy_;

  bool _running;
};
}
