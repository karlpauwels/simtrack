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

namespace pose {

void computeResidualFlow(float *d_res_flowx, float *d_res_flowy,
                         float *d_res_ar_flowx, float *d_res_ar_flowy,
                         const float *d_flowx, const float *d_flowy,
                         const float *d_ar_flowx, const float *d_ar_flowy,
                         const float *d_delta_T_accum,
                         const float *d_delta_Rmat_accum, const float *d_init_Z,
                         const cudaArray *d_segment_ind, int n_cols, int n_rows,
                         float nodal_point_x, float nodal_point_y,
                         float focal_length_x, float focal_length_y);

void markValidFlowZbufferAndZbufferZeroBased(
    unsigned int *d_valid_ar_flow_Zbuffer, unsigned int *d_valid_Zbuffer,
    const float *d_ar_flowx, const cudaArray *d_segmentINDArray, int n_cols,
    int n_rows, int n_objects);

void mark_with_zero_based_segmentIND(
    unsigned int *d_valid_flow_Zbuffer, unsigned int *d_valid_disparity_Zbuffer,
    const float *d_flowx, const float *d_ar_flowx, const char *d_disparity,
    const cudaArray *d_segmentINDArray, int n_cols, int n_rows, int n_objects,
    int d_disparity_pitch, bool mark_flow, bool mark_ar_flow,
    bool mark_disparity, int segments_to_update);

void subsample_ind_and_labels(int *d_ind_sub, const int *d_ind,
                              unsigned int *d_label_sub,
                              const unsigned int *d_label, int n_out,
                              float inv_sub_factor);

void gather_valid_flow_Zbuffer(float2 *d_flow_compact, float *d_Zbuffer_compact,
                               const float *d_flowx, const float *d_flowy,
                               const float *d_ar_flowx, const float *d_ar_flowy,
                               int *d_ind_flow_Zbuffer,
                               const cudaArray *d_ZbufferArray,
                               int n_valid_flow_Zbuffer, int n_cols, int n_rows,
                               float Z_conv1, float Z_conv2,
                               int ind_flow_offset = 0);

void gather_valid_disparity_Zbuffer(
    float *d_disparity_compact, float4 *d_Zbuffer_normals_compact,
    const char *d_disparity, int *d_ind_disparity_Zbuffer,
    const cudaArray *d_ZbufferArray, const cudaArray *d_normalXArray,
    const cudaArray *d_normalYArray, const cudaArray *d_normalZArray,
    int n_valid_disparity_Zbuffer, int n_cols, int n_rows, float Z_conv1,
    float Z_conv2, int disparity_pitch, int ind_disp_offset = 0);

void normal_eqs_flow(dim3 blockGrid, dim3 threadBlock, float *d_CO,
                     const float2 *d_flow_compact,
                     const float *d_Zbuffer_flow_compact,
                     const int *d_ind_flow_Zbuffer, float fx, float fy,
                     float ox, float oy, int n_rows, int n_cols,
                     const int *d_n_values_flow, const int *d_start_ind_flow);

void normal_eqs_disparity(dim3 blockGrid, dim3 threadBlock, float *d_CD,
                          const float *d_disparity_compact,
                          const float4 *d_Zbuffer_normals_compact,
                          const int *d_ind_disparity_Zbuffer, float fx,
                          float fy, float ox, float oy, float b, int n_cols,
                          const int *d_n_values_disparity,
                          const int *d_start_ind_disparity, float w_disp);

void reduce_normal_eqs_64_mult_constr(dim3 blockGrid, dim3 threadBlock,
                                      float *d_C_reduced, const float *d_C,
                                      int gridDim_x_normal_equations,
                                      int n_constraints);

void flow_absolute_residual_scalable(
    dim3 blockGrid, dim3 threadBlock, float *d_abs_res,
    const float2 *d_flow_compact, const float *d_Zbuffer_flow_compact,
    const int *d_ind_flow_Zbuffer, const unsigned int *d_valid_flow_Zbuffer,
    float fx, float fy, float ox, float oy, int n_rows, int n_cols,
    int n_valid_flow_Zbuffer, const int *d_offset_ind,
    const int *d_segment_translation_table, float w_flow, float w_ar_flow,
    const float *d_dTR);

void disp_absolute_residual_scalable(
    dim3 blockGrid, dim3 threadBlock, float *d_abs_res,
    const float *d_disparity_compact, const float4 *d_Zbuffer_normals_compact,
    const int *d_ind_disparity_Zbuffer,
    const unsigned int *d_valid_disparity_Zbuffer, float fx, float fy, float ox,
    float oy, float b, int n_cols, int n_valid_disparity_Zbuffer,
    const int *d_offset_ind, const int *d_segment_translation_table,
    float w_disp, const float *d_dTR);

void normal_eqs_flow_weighted(dim3 blockGrid, dim3 threadBlock, float *d_CO,
                              const float2 *d_flow_compact,
                              const float *d_Zbuffer_flow_compact,
                              const int *d_ind_flow_Zbuffer, float fx, float fy,
                              float ox, float oy, int n_rows, int n_cols,
                              const int *d_n_values_flow,
                              const int *d_start_ind_flow,
                              const float *d_abs_res_scales, float w_flow,
                              float w_ar_flow, const float *d_dTR);

void normal_eqs_disparity_weighted(
    dim3 blockGrid, dim3 threadBlock, float *d_CD,
    const float *d_disparity_compact, const float4 *d_Zbuffer_normals_compact,
    const int *d_ind_disparity_Zbuffer, float fx, float fy, float ox, float oy,
    float b, int n_cols, const int *d_n_values_disparity,
    const int *d_start_ind_disparity, const float *d_abs_res_scales,
    float w_disp, const float *d_dTR);
}
