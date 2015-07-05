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

#include <cstdio>
#include <vector>
#include <bitset>
#include <utility_kernels_pose.h>
#include <multiple_rigid_pose_kernels.h>

namespace pose {

// OpenGL mapped input textures
texture<float, cudaTextureType2D, cudaReadModeElementType> d_Zbuffer_texture;
texture<float, cudaTextureType2D, cudaReadModeElementType>
d_normalXArray_texture;
texture<float, cudaTextureType2D, cudaReadModeElementType>
d_normalYArray_texture;
texture<float, cudaTextureType2D, cudaReadModeElementType>
d_normalZArray_texture;
texture<float, cudaTextureType2D, cudaReadModeElementType>
d_segmentINDArray_texture;

// Residual flow after compensating for d_T and d_R_mat applied to
// d_init_Zbuffer

__global__ void
compute_residual_flow_GPU(float *d_res_flowx, float *d_res_flowy,
                          const float *d_flowx, const float *d_flowy,
                          const float *d_T, const float *d_R_mat,
                          const float *d_init_Z, int n_cols, int n_rows,
                          float ox, float oy, float fx, float fy) {

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < n_cols) & (y < n_rows)) // are we in the image?
  {
    unsigned int ind = x + y * n_cols;

    float ux = d_flowx[ind];

    // determine gl coord
    float x_gl = (float)x + 0.5f;
    float y_gl = (float)y + 0.5f;

    int segment = (int)rintf(tex2D(d_segmentINDArray_texture, x_gl, y_gl));

    if (isfinite(ux) & (segment > 0)) { // check validity

      // move T and R pointers to correct position
      d_T += 3 * segment - 3;
      d_R_mat += 9 * segment - 9;

      float uy = d_flowy[ind];
      float Z = d_init_Z[ind];

      float xt = __fdividef((x - ox), fx);
      float yt = __fdividef((y - oy), fy);

      // reconstruct initial model point
      float X = xt * Z;
      float Y = yt * Z;

      // rigid transform model point
      float X2 = d_R_mat[0] * X + d_R_mat[1] * Y + d_R_mat[2] * Z + d_T[0];
      float Y2 = d_R_mat[3] * X + d_R_mat[4] * Y + d_R_mat[5] * Z + d_T[1];
      float Z2 = d_R_mat[6] * X + d_R_mat[7] * Y + d_R_mat[8] * Z + d_T[2];

      // explained flow
      float ux_e = fx * X2 / Z2 - fx * xt;
      float uy_e = fy * Y2 / Z2 - fy * yt;

      // save residual flow
      d_res_flowx[ind] = ux - ux_e;
      d_res_flowy[ind] = uy - uy_e;

    } else {
      d_res_flowx[ind] = nanf("");
      d_res_flowy[ind] = nanf("");
    }
  }
}

// Marks the valid flow locations with (segment index - 1) and the invalids with
// (n_objects) and also just the zbuffer locations
// --> invalids will be sorted after all valids

__global__ void mark_valid_flow_Zbuffer_and_Zbuffer_zero_based_GPU(
    unsigned int *d_valid_flow_Zbuffer, unsigned int *d_valid_Zbuffer,
    const float *d_flowx, int n_cols, int n_rows, int n_objects) {

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < n_cols) & (y < n_rows)) // are we in the image?
  {
    // determine linear index
    unsigned int ind = x + y * n_cols;

    // determine gl coord
    float xt = (float)x + 0.5f;
    float yt = (float)y + 0.5f;

    // fetch segment index
    int segmentIND = (int)rintf(tex2D(d_segmentINDArray_texture, xt, yt));

    // change to zero-based index, with invalids = n_objects
    segmentIND = (segmentIND == 0) ? n_objects : (segmentIND - 1);

    d_valid_Zbuffer[ind] = segmentIND;
    d_valid_flow_Zbuffer[ind] =
        (isfinite(d_flowx[ind])) ? segmentIND : n_objects;
  }
}

// Marks the valid flow and disparity locations that also belong to a specific
// segment, with that segment's index - 1
// The returned segment indices are zero-based and invalids obtain label
// n_objects
// as a result, invalids will be sorted after all valids
// Segments with 0 in the (zero-based) binary representation of
// segments_to_update
// are considered invalid
__global__ void mark_with_zero_based_segmentIND_GPU(
    unsigned int *d_valid_flow_Zbuffer, unsigned int *d_valid_disparity_Zbuffer,
    const float *d_flowx, const float *d_ar_flowx, const char *d_disparity,
    int n_cols, int n_rows, int n_objects, int d_disparity_pitch,
    bool mark_flow, bool mark_ar_flow, bool mark_disparity,
    int segments_to_update) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < n_cols) & (y < n_rows)) // are we in the image?
  {
    // determine linear index
    unsigned int ind = x + y * n_cols;

    // determine gl coord
    float xt = (float)x + 0.5f;
    float yt = (float)y + 0.5f;

    // fetch segment index
    int segmentIND = (int)rintf(tex2D(d_segmentINDArray_texture, xt, yt));

    // change to zero-based index, with invalids = n_objects
    //    segmentIND = ( (segmentIND==0) || !(segments_to_update & (1 <<
    // (segmentIND-1))) ) ? n_objects : (segmentIND-1);

    segmentIND = (segmentIND == 0) ? n_objects : (segmentIND - 1);

    // mark flow
    d_valid_flow_Zbuffer[ind] =
        (isfinite(d_flowx[ind]) && mark_flow) ? segmentIND : n_objects;

    // ar flow is marked at index + n_cols*n_rows
    d_valid_flow_Zbuffer[ind + n_cols * n_rows] =
        (isfinite(d_ar_flowx[ind]) && mark_ar_flow) ? segmentIND : n_objects;

    // fetch disparity
    float *disparity = (float *)(d_disparity + y * d_disparity_pitch) + x;
    d_valid_disparity_Zbuffer[ind] =
        (isfinite(disparity[0]) && mark_disparity) ? segmentIND : n_objects;
  }
}

// Marks the valid flow and disparity locations that also belong to a specific
// segment, with that segment's index - 1 + index_offset
// The returned segment indices are zero-based and invalids obtain label
// 'invalid_index'
// as a result, invalids will be sorted after all valids
__global__ void mark_flow_disparity_GPU(
    unsigned int *d_valid_flow_Zbuffer, unsigned int *d_valid_disparity_Zbuffer,
    const float *d_flowx, const float *d_ar_flowx, const char *d_disparity,
    int n_cols, int n_rows, int invalid_index, int index_offset,
    int d_disparity_pitch, bool mark_flow, bool mark_ar_flow,
    bool mark_disparity) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < n_cols) & (y < n_rows)) // are we in the image?
  {
    // determine linear index
    unsigned int ind = x + y * n_cols;

    // determine gl coord
    float xt = (float)x + 0.5f;
    float yt = (float)y + 0.5f;

    // fetch segment index
    int segmentIND = (int)rintf(tex2D(d_segmentINDArray_texture, xt, yt));

    // change to zero-based index (with offset), with invalids = invalid_index
    segmentIND =
        (segmentIND == 0) ? invalid_index : (segmentIND - 1 + index_offset);

    // mark flow
    d_valid_flow_Zbuffer[ind] =
        (isfinite(d_flowx[ind]) && mark_flow) ? segmentIND : invalid_index;

    // ar flow is marked at index + n_cols*n_rows
    d_valid_flow_Zbuffer[ind + n_cols * n_rows] =
        (isfinite(d_ar_flowx[ind]) && mark_ar_flow) ? segmentIND
                                                    : invalid_index;

    // fetch disparity
    float *disparity = (float *)(d_disparity + y * d_disparity_pitch) + x;
    d_valid_disparity_Zbuffer[ind] =
        (isfinite(disparity[0]) && mark_disparity) ? segmentIND : invalid_index;
  }
}

// Marks the valid flow locations that also belong to a specific segment, with
// that segment's index - 1 + index_offset
// The returned segment indices are zero-based and invalids obtain label
// 'invalid_index'
// as a result, invalids will be sorted after all valids
__global__ void mark_flow_GPU(unsigned int *d_valid_flow_Zbuffer,
                              const float *d_flowx, const float *d_ar_flowx,
                              int n_cols, int n_rows, int invalid_index,
                              int index_offset, bool mark_flow,
                              bool mark_ar_flow) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < n_cols) & (y < n_rows)) // are we in the image?
  {
    // determine linear index
    unsigned int ind = x + y * n_cols;

    // determine gl coord
    float xt = (float)x + 0.5f;
    float yt = (float)y + 0.5f;

    // fetch segment index
    int segmentIND = (int)rintf(tex2D(d_segmentINDArray_texture, xt, yt));

    // change to zero-based index (with offset), with invalids = invalid_index
    segmentIND =
        (segmentIND == 0) ? invalid_index : (segmentIND - 1 + index_offset);

    // mark flow
    d_valid_flow_Zbuffer[ind] =
        (isfinite(d_flowx[ind]) && mark_flow) ? segmentIND : invalid_index;

    // ar flow is marked at index + n_cols*n_rows
    d_valid_flow_Zbuffer[ind + n_cols * n_rows] =
        (isfinite(d_ar_flowx[ind]) && mark_ar_flow) ? segmentIND
                                                    : invalid_index;
  }
}

// Regularly subsample indices and labels
__global__ void subsample_ind_and_labels_GPU(int *d_ind_sub, const int *d_ind,
                                             unsigned int *d_label_sub,
                                             const unsigned int *d_label,
                                             int n_out, float inv_sub_factor) {

  unsigned int ind_out = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind_out < n_out) {

    int ind_in = (int)floorf((float)(ind_out) * inv_sub_factor);
    d_ind_sub[ind_out] = d_ind[ind_in];
    d_label_sub[ind_out] = d_label[ind_in];
  }
}

// Gather the valid flow and Zbuffer + transform Zbuffer
// ind_flow_offset used in multi-camera case
__global__ void gather_valid_flow_Zbuffer_GPU(
    float2 *d_flow_compact, float *d_Zbuffer_compact, const float *d_flowx,
    const float *d_flowy, const float *d_ar_flowx, const float *d_ar_flowy,
    const int *d_ind_flow_Zbuffer, int n_valid_flow_Zbuffer, int n_cols,
    int n_rows, float Z_conv1, float Z_conv2, int ind_flow_offset = 0) {
  unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n_valid_flow_Zbuffer) {

    int ind_flow = d_ind_flow_Zbuffer[ind] - ind_flow_offset;

    // fetch and write flow
    if (ind_flow < (n_rows * n_cols)) // image flow
      d_flow_compact[ind] = make_float2(d_flowx[ind_flow], d_flowy[ind_flow]);
    else { // ar flow
      ind_flow -= n_rows * n_cols;
      d_flow_compact[ind] =
          make_float2(d_ar_flowx[ind_flow], d_ar_flowy[ind_flow]);
    }

    // extract row (y) and column (x) from linear index
    int y = floorf(__fdividef((float)ind_flow, n_cols));
    int x = ind_flow - y * n_cols;

    // determine gl index
    float xt = (float)x + 0.5f;
    float yt = (float)y + 0.5f;

    // fetch Zbuffer
    float Zbuffer = tex2D(d_Zbuffer_texture, xt, yt);

    // transform Zbuffer
    Zbuffer = __fdividef(Z_conv1, Zbuffer + Z_conv2);

    // write Zbuffer
    d_Zbuffer_compact[ind] = Zbuffer;
  }
}

// Gather the valid disparity, Zbuffer and normals + transform Zbuffer and
// normals
__global__ void gather_valid_disparity_Zbuffer_GPU(
    float *d_disparity_compact, float4 *d_Zbuffer_normals_compact,
    const char *d_disparity, const int *d_ind_disparity_Zbuffer,
    int n_valid_disparity_Zbuffer, int n_cols, int n_rows, float Z_conv1,
    float Z_conv2, int disparity_pitch, int ind_disp_offset = 0) {
  unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n_valid_disparity_Zbuffer) {

    int ind_disp = d_ind_disparity_Zbuffer[ind] - ind_disp_offset;

    // extract row (y) and column (x) from linear index
    int y = floorf(__fdividef((float)ind_disp, n_cols));
    int x = ind_disp - y * n_cols;

    // fetch and write disparity
    d_disparity_compact[ind] =
        *((float *)(d_disparity + y * disparity_pitch) + x);

    // determine gl index
    float xt = (float)x + 0.5f;
    float yt = (float)y + 0.5f;

    // fetch Zbuffer and normal
    float Zbuffer = tex2D(d_Zbuffer_texture, xt, yt);
    float normalx = tex2D(d_normalXArray_texture, xt, yt);
    float normaly = tex2D(d_normalYArray_texture, xt, yt);
    float normalz = tex2D(d_normalZArray_texture, xt, yt);

    // transform Zbuffer
    Zbuffer = __fdividef(Z_conv1, Zbuffer + Z_conv2);

    // transform normal
    normalz = -normalz;

    // write Zbuffer and normal
    d_Zbuffer_normals_compact[ind] =
        make_float4(Zbuffer, normalx, normaly, normalz);
  }
}

// Unweighted normal equations for flow

__global__ void normal_eqs_flow_GPU(float *d_CO, const float2 *d_flow_compact,
                                    const float *d_Zbuffer_flow_compact,
                                    const int *d_ind_flow_Zbuffer, float fx,
                                    float fy, float ox, float oy, int n_rows,
                                    int n_cols, const int *d_n_values_flow,
                                    const int *d_start_ind_flow) {

  int n_val_accum = gridDim.x * blockDim.x; // _MAX_N_VAL_ACCUM may not be
                                            // multiple of blocksize

  int n_flow = d_n_values_flow[blockIdx.y];
  int n_accum = (int)ceilf((float)n_flow / (float)n_val_accum);
  int start_ind = d_start_ind_flow[blockIdx.y];

  // initialize accumulators

  float A0 = 0.0f, A1 = 0.0f, A2 = 0.0f, A3 = 0.0f, A4 = 0.0f, A5 = 0.0f,
        A6 = 0.0f, A7 = 0.0f, A8 = 0.0f, A9 = 0.0f, A10 = 0.0f, A11 = 0.0f,
        A12 = 0.0f, A13 = 0.0f, A14 = 0.0f, A15 = 0.0f, A16 = 0.0f, A17 = 0.0f,
        A18 = 0.0f, A19 = 0.0f, A20 = 0.0f, A21 = 0.0f, A22 = 0.0f;

  for (int in_ind = blockDim.x * blockIdx.x * n_accum + threadIdx.x;
       in_ind < blockDim.x * (blockIdx.x + 1) * n_accum; in_ind += blockDim.x) {

    if (in_ind < n_flow) { // is this a valid sample?

      // fetch flow and Zbuffer from global memory
      float2 u = d_flow_compact[in_ind + start_ind];
      float disp = __fdividef(1.0f, d_Zbuffer_flow_compact[in_ind + start_ind]);

      // compute coordinates
      int pixel_ind = d_ind_flow_Zbuffer[in_ind + start_ind];
      bool is_ar_flow = (pixel_ind >= (n_rows * n_cols));
      pixel_ind -= (int)is_ar_flow * n_rows * n_cols;

      float y = floorf(__fdividef((float)pixel_ind, n_cols));
      float x = (float)pixel_ind - y * n_cols;

      x = x - ox;
      y = y - oy;

      /************************/
      /* evaluate constraints */
      /************************/

      // unique values A-matrix
      A0 += (disp * disp * fx * fx);
      A1 += (-disp * disp * x * fx);
      A2 += (-disp * x * y);
      A3 += (disp * fx * fx + disp * x * x);
      A4 += (-disp * y * fx);
      A5 += (-disp * disp * y * fy);
      A6 += (-disp * fy * fy - disp * y * y); //!!!!
      A7 += (disp * x * fy);
      A8 += (disp * disp * x * x + disp * disp * y * y);
      A9 += (disp * x * x * y / fx + disp * y * fy + disp * y * y * y / fy);
      A10 += (-disp * x * fx - disp * x * x * x / fx - disp * x * y * y / fy);
      A11 += (x * x * y * y / (fx * fx) + fy * fy + 2.0f * y * y +
              y * y * y * y / (fy * fy));
      A12 += (-2.0f * x * y - x * x * x * y / (fx * fx) -
              x * y * y * y / (fy * fy));
      A13 += (x * y * y / fx - x * fy - x * y * y / fy);
      A14 += (fx * fx + 2.0f * x * x + x * x * x * x / (fx * fx) +
              x * x * y * y / (fy * fy));
      A15 += (-y * fx - x * x * y / fx + x * x * y / fy);
      A16 += (x * x + y * y);

      // B-vector

      A17 += (disp * u.x * fx);
      A18 += (disp * u.y * fy);
      A19 += (-disp * x * u.x - disp * y * u.y);
      A20 += (-x * y * u.x / fx - u.y * fy - u.y * y * y / fy);
      A21 += (u.x * fx + x * x * u.x / fx + x * y * u.y / fy);
      A22 += (-y * u.x + x * u.y);
    }
  }

  /**************************/
  /* write out accumulators */
  /**************************/

  int out_ind =
      23 * n_val_accum * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

  d_CO[out_ind] = A0;
  d_CO[out_ind + n_val_accum] = A1;
  d_CO[out_ind + 2 * n_val_accum] = A2;
  d_CO[out_ind + 3 * n_val_accum] = A3;
  d_CO[out_ind + 4 * n_val_accum] = A4;
  d_CO[out_ind + 5 * n_val_accum] = A5;
  d_CO[out_ind + 6 * n_val_accum] = A6;
  d_CO[out_ind + 7 * n_val_accum] = A7;
  d_CO[out_ind + 8 * n_val_accum] = A8;
  d_CO[out_ind + 9 * n_val_accum] = A9;
  d_CO[out_ind + 10 * n_val_accum] = A10;
  d_CO[out_ind + 11 * n_val_accum] = A11;
  d_CO[out_ind + 12 * n_val_accum] = A12;
  d_CO[out_ind + 13 * n_val_accum] = A13;
  d_CO[out_ind + 14 * n_val_accum] = A14;
  d_CO[out_ind + 15 * n_val_accum] = A15;
  d_CO[out_ind + 16 * n_val_accum] = A16;
  d_CO[out_ind + 17 * n_val_accum] = A17;
  d_CO[out_ind + 18 * n_val_accum] = A18;
  d_CO[out_ind + 19 * n_val_accum] = A19;
  d_CO[out_ind + 20 * n_val_accum] = A20;
  d_CO[out_ind + 21 * n_val_accum] = A21;
  d_CO[out_ind + 22 * n_val_accum] = A22;
}

// Unweighted normal equations for flow in multicamera scenario with different
// calibration parameters and pixel offsets in d_ind_disp_Zbuffer

__global__ void normal_eqs_flow_multicam_GPU(
    float *d_CO, float2 *d_flow_compact, float *d_Zbuffer_flow_compact,
    int *d_ind_flow_Zbuffer, const float *d_focal_length,
    const float *d_nodal_point_x, const float *d_nodal_point_y,
    const int *d_n_rows, const int *d_n_cols, const int *d_n_values_flow,
    const int *d_start_ind_flow, const int *d_pixel_ind_offset) {
  int n_val_accum = gridDim.x * blockDim.x; // _MAX_N_VAL_ACCUM may not be
                                            // multiple of blocksize

  int n_flow = d_n_values_flow[blockIdx.y];
  int n_accum = (int)ceilf((float)n_flow / (float)n_val_accum);
  int start_ind = d_start_ind_flow[blockIdx.y];

  float f = d_focal_length[blockIdx.y];
  float ox = d_nodal_point_x[blockIdx.y];
  float oy = d_nodal_point_y[blockIdx.y];
  int n_rows = d_n_rows[blockIdx.y];
  int n_cols = d_n_cols[blockIdx.y];
  int pixel_ind_offset = d_pixel_ind_offset[blockIdx.y];

  // initialize accumulators

  float A0 = 0.0f, A1 = 0.0f, A2 = 0.0f, A3 = 0.0f, A4 = 0.0f, A5 = 0.0f,
        A6 = 0.0f, A7 = 0.0f, A8 = 0.0f, A9 = 0.0f, A10 = 0.0f, A11 = 0.0f,
        A12 = 0.0f, A13 = 0.0f, A14 = 0.0f, A15 = 0.0f, A16 = 0.0f, A17 = 0.0f,
        A18 = 0.0f, A19 = 0.0f, A20 = 0.0f, A21 = 0.0f, A22 = 0.0f;

  for (int in_ind = blockDim.x * blockIdx.x * n_accum + threadIdx.x;
       in_ind < blockDim.x * (blockIdx.x + 1) * n_accum; in_ind += blockDim.x) {

    if (in_ind < n_flow) { // is this a valid sample?

      // fetch flow and Zbuffer from global memory
      float2 u = d_flow_compact[in_ind + start_ind];
      float disp = __fdividef(1.0f, d_Zbuffer_flow_compact[in_ind + start_ind]);

      // compute coordinates
      int pixel_ind = d_ind_flow_Zbuffer[in_ind + start_ind] - pixel_ind_offset;
      bool is_ar_flow = (pixel_ind >= (n_rows * n_cols));
      pixel_ind -= (int)is_ar_flow * n_rows * n_cols;

      float y = floorf(__fdividef((float)pixel_ind, n_cols));
      float x = (float)pixel_ind - y * n_cols;

      x = x - ox;
      y = y - oy;

      // flip y axis
      y = -y;
      u.y = -u.y;

      /************************/
      /* evaluate constraints */
      /************************/

      // unique values A-matrix
      A0 += (disp * disp * f * f);
      A1 += (-disp * disp * x * f);
      A2 += (-disp * x * y);
      A3 += (disp * f * f + disp * x * x);
      A4 += (-disp * y * f);
      A5 += (-disp * disp * y * f);
      A6 += (-disp * f * f - disp * y * y);
      A7 += (disp * x * f);
      A8 += (disp * disp * x * x + disp * disp * y * y);
      A9 += (disp * x * x * y / f + disp * y * f + disp * y * y * y / f);
      A10 += (-disp * x * f - disp * x * x * x / f - disp * x * y * y / f);
      A11 += (x * x * y * y / (f * f) + f * f + 2.0f * y * y +
              y * y * y * y / (f * f));
      A12 +=
          (-2.0f * x * y - x * x * x * y / (f * f) - x * y * y * y / (f * f));
      A13 += (-x * f);
      A14 += (f * f + 2.0f * x * x + x * x * x * x / (f * f) +
              x * x * y * y / (f * f));
      A15 += (-y * f);
      A16 += (x * x + y * y);

      // B-vector

      A17 += (disp * u.x * f);
      A18 += (disp * u.y * f);
      A19 += (-disp * x * u.x - disp * y * u.y);
      A20 += (-x * y * u.x / f - u.y * f - u.y * y * y / f);
      A21 += (u.x * f + x * x * u.x / f + x * y * u.y / f);
      A22 += (-y * u.x + x * u.y);
    }
  }

  /**************************/
  /* write out accumulators */
  /**************************/

  int out_ind =
      23 * n_val_accum * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

  d_CO[out_ind] = A0;
  d_CO[out_ind + n_val_accum] = A1;
  d_CO[out_ind + 2 * n_val_accum] = A2;
  d_CO[out_ind + 3 * n_val_accum] = A3;
  d_CO[out_ind + 4 * n_val_accum] = A4;
  d_CO[out_ind + 5 * n_val_accum] = A5;
  d_CO[out_ind + 6 * n_val_accum] = A6;
  d_CO[out_ind + 7 * n_val_accum] = A7;
  d_CO[out_ind + 8 * n_val_accum] = A8;
  d_CO[out_ind + 9 * n_val_accum] = A9;
  d_CO[out_ind + 10 * n_val_accum] = A10;
  d_CO[out_ind + 11 * n_val_accum] = A11;
  d_CO[out_ind + 12 * n_val_accum] = A12;
  d_CO[out_ind + 13 * n_val_accum] = A13;
  d_CO[out_ind + 14 * n_val_accum] = A14;
  d_CO[out_ind + 15 * n_val_accum] = A15;
  d_CO[out_ind + 16 * n_val_accum] = A16;
  d_CO[out_ind + 17 * n_val_accum] = A17;
  d_CO[out_ind + 18 * n_val_accum] = A18;
  d_CO[out_ind + 19 * n_val_accum] = A19;
  d_CO[out_ind + 20 * n_val_accum] = A20;
  d_CO[out_ind + 21 * n_val_accum] = A21;
  d_CO[out_ind + 22 * n_val_accum] = A22;
}

// Unweighted normal equations for disparity

__global__ void
normal_eqs_disparity_GPU(float *d_CD, const float *d_disparity_compact,
                         const float4 *d_Zbuffer_normals_compact,
                         const int *d_ind_disparity_Zbuffer, float fx, float fy,
                         float ox, float oy, float b, int n_cols,
                         const int *d_n_values_disparity,
                         const int *d_start_ind_disparity, float w_disp) {

  int n_val_accum = gridDim.x * blockDim.x; // _MAX_N_VAL_ACCUM may not be
                                            // multiple of blocksize

  int n_disparity = d_n_values_disparity[blockIdx.y];
  int n_accum = (int)ceilf((float)n_disparity / (float)n_val_accum);
  int start_ind = d_start_ind_disparity[blockIdx.y];

  // initialize accumulators

  float A0 = 0.0f, A1 = 0.0f, A2 = 0.0f, A3 = 0.0f, A4 = 0.0f, A5 = 0.0f,
        A6 = 0.0f, A7 = 0.0f, A8 = 0.0f, A9 = 0.0f, A10 = 0.0f, A11 = 0.0f,
        A12 = 0.0f, A13 = 0.0f, A14 = 0.0f, A15 = 0.0f, A16 = 0.0f, A17 = 0.0f,
        A18 = 0.0f, A19 = 0.0f, A20 = 0.0f, A21 = 0.0f, A22 = 0.0f, A23 = 0.0f,
        A24 = 0.0f, A25 = 0.0f, A26 = 0.0f;

  for (int in_ind = blockDim.x * blockIdx.x * n_accum + threadIdx.x;
       in_ind < blockDim.x * (blockIdx.x + 1) * n_accum; in_ind += blockDim.x) {

    if (in_ind < n_disparity) { // is this a valid sample?

      // fetch disparity, Zbuffer and normal from global memory
      float disp = d_disparity_compact[in_ind + start_ind];
      float4 tmp = d_Zbuffer_normals_compact[in_ind + start_ind];
      float Zbuffer = tmp.x;
      float nx = tmp.y;
      float ny = tmp.z;
      float nz = tmp.w;

      // compute coordinates
      int pixel_ind = d_ind_disparity_Zbuffer[in_ind + start_ind];

      float y = floorf(__fdividef((float)pixel_ind, n_cols));
      float x = (float)pixel_ind - y * n_cols;

      x = __fdividef((x - ox), fx);
      y = __fdividef((y - oy), fy);

      // reconstruct 3D point from disparity

      float Zd = -(fx * b) / disp; // arbitrary conversion for now using fx
      float Xd = x * Zd;
      float Yd = y * Zd;

      // reconstruct 3D point from model

      float Zm = Zbuffer;
      float Xm = x * Zm;
      float Ym = y * Zm;

      // weight the constraint according to (fx*b)/(Zm*Zm) to convert
      // from distance- (mm) to image-units (pixel)
      float w2 = fx * b / (Zm * Zm);
      w2 *= w2;

      /************************/
      /* evaluate constraints */
      /************************/

      // unique values A-matrix

      A0 += w2 * (nx * nx);
      A1 += w2 * (nx * ny);
      A2 += w2 * (nx * nz);
      A3 += w2 * (Ym * nx * nz - Zm * nx * ny);
      A4 += w2 * (Zm * (nx * nx) - Xm * nx * nz);
      A5 += w2 * (-Ym * (nx * nx) + Xm * nx * ny);

      A6 += w2 * (ny * ny);
      A7 += w2 * (ny * nz);
      A8 += w2 * (-Zm * (ny * ny) + Ym * ny * nz);
      A9 += w2 * (-Xm * ny * nz + Zm * nx * ny);
      A10 += w2 * (Xm * (ny * ny) - Ym * nx * ny);

      A11 += w2 * (nz * nz);
      A12 += w2 * (Ym * (nz * nz) - Zm * ny * nz);
      A13 += w2 * (-Xm * (nz * nz) + Zm * nx * nz);
      A14 += w2 * (Xm * ny * nz - Ym * nx * nz);

      A15 += w2 * ((Ym * Ym) * (nz * nz) + (Zm * Zm) * (ny * ny) -
                   Ym * Zm * ny * nz * 2.0f);
      A16 += w2 * (-Xm * Ym * (nz * nz) - (Zm * Zm) * nx * ny +
                   Xm * Zm * ny * nz + Ym * Zm * nx * nz);
      A17 += w2 * (-Xm * Zm * (ny * ny) - (Ym * Ym) * nx * nz +
                   Xm * Ym * ny * nz + Ym * Zm * nx * ny);

      A18 += w2 * ((Xm * Xm) * (nz * nz) + (Zm * Zm) * (nx * nx) -
                   Xm * Zm * nx * nz * 2.0f);
      A19 += w2 * (-Ym * Zm * (nx * nx) - (Xm * Xm) * ny * nz +
                   Xm * Ym * nx * nz + Xm * Zm * nx * ny);

      A20 += w2 * ((Xm * Xm) * (ny * ny) + (Ym * Ym) * (nx * nx) -
                   Xm * Ym * nx * ny * 2.0f);

      // B-vector

      A21 += w2 * (Xd * (nx * nx) - Xm * (nx * nx) + Yd * nx * ny -
                   Ym * nx * ny + Zd * nx * nz - Zm * nx * nz);
      A22 += w2 * (Yd * (ny * ny) - Ym * (ny * ny) + Xd * nx * ny -
                   Xm * nx * ny + Zd * ny * nz - Zm * ny * nz);
      A23 += w2 * (Zd * (nz * nz) - Zm * (nz * nz) + Xd * nx * nz -
                   Xm * nx * nz + Yd * ny * nz - Ym * ny * nz);
      A24 += w2 *
             (-Yd * Zm * (ny * ny) + Ym * Zd * (nz * nz) + Ym * Zm * (ny * ny) -
              Ym * Zm * (nz * nz) - (Ym * Ym) * ny * nz + (Zm * Zm) * ny * nz +
              Xd * Ym * nx * nz - Xm * Ym * nx * nz - Xd * Zm * nx * ny +
              Yd * Ym * ny * nz + Xm * Zm * nx * ny - Zd * Zm * ny * nz);
      A25 += w2 *
             (Xd * Zm * (nx * nx) - Xm * Zd * (nz * nz) - Xm * Zm * (nx * nx) +
              Xm * Zm * (nz * nz) + (Xm * Xm) * nx * nz - (Zm * Zm) * nx * nz -
              Xd * Xm * nx * nz - Xm * Yd * ny * nz + Xm * Ym * ny * nz +
              Yd * Zm * nx * ny - Ym * Zm * nx * ny + Zd * Zm * nx * nz);
      A26 += w2 *
             (-Xd * Ym * (nx * nx) + Xm * Yd * (ny * ny) + Xm * Ym * (nx * nx) -
              Xm * Ym * (ny * ny) - (Xm * Xm) * nx * ny + (Ym * Ym) * nx * ny +
              Xd * Xm * nx * ny - Yd * Ym * nx * ny + Xm * Zd * ny * nz -
              Xm * Zm * ny * nz - Ym * Zd * nx * nz + Ym * Zm * nx * nz);
    }
  }

  /**************************/
  /* write out accumulators */
  /**************************/

  int out_ind =
      27 * n_val_accum * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

  w_disp *= w_disp; // weight relative to flow

  d_CD[out_ind] = w_disp * A0;
  d_CD[out_ind + n_val_accum] = w_disp * A1;
  d_CD[out_ind + 2 * n_val_accum] = w_disp * A2;
  d_CD[out_ind + 3 * n_val_accum] = w_disp * A3;
  d_CD[out_ind + 4 * n_val_accum] = w_disp * A4;
  d_CD[out_ind + 5 * n_val_accum] = w_disp * A5;
  d_CD[out_ind + 6 * n_val_accum] = w_disp * A6;
  d_CD[out_ind + 7 * n_val_accum] = w_disp * A7;
  d_CD[out_ind + 8 * n_val_accum] = w_disp * A8;
  d_CD[out_ind + 9 * n_val_accum] = w_disp * A9;
  d_CD[out_ind + 10 * n_val_accum] = w_disp * A10;
  d_CD[out_ind + 11 * n_val_accum] = w_disp * A11;
  d_CD[out_ind + 12 * n_val_accum] = w_disp * A12;
  d_CD[out_ind + 13 * n_val_accum] = w_disp * A13;
  d_CD[out_ind + 14 * n_val_accum] = w_disp * A14;
  d_CD[out_ind + 15 * n_val_accum] = w_disp * A15;
  d_CD[out_ind + 16 * n_val_accum] = w_disp * A16;
  d_CD[out_ind + 17 * n_val_accum] = w_disp * A17;
  d_CD[out_ind + 18 * n_val_accum] = w_disp * A18;
  d_CD[out_ind + 19 * n_val_accum] = w_disp * A19;
  d_CD[out_ind + 20 * n_val_accum] = w_disp * A20;
  d_CD[out_ind + 21 * n_val_accum] = w_disp * A21;
  d_CD[out_ind + 22 * n_val_accum] = w_disp * A22;
  d_CD[out_ind + 23 * n_val_accum] = w_disp * A23;
  d_CD[out_ind + 24 * n_val_accum] = w_disp * A24;
  d_CD[out_ind + 25 * n_val_accum] = w_disp * A25;
  d_CD[out_ind + 26 * n_val_accum] = w_disp * A26;
}

// Unweighted normal equations for disparity in multicamera scenario with
// different calibration parameters and pixel offsets in d_ind_flow_Zbuffer

__global__ void normal_eqs_disparity_multicam_GPU(
    float *d_CD, float *d_disparity_compact, float4 *d_Zbuffer_normals_compact,
    int *d_ind_disparity_Zbuffer, const float *d_focal_length,
    const float *d_nodal_point_x, const float *d_nodal_point_y,
    const float *d_baseline, const int *d_n_cols,
    const int *d_n_values_disparity, const int *d_start_ind_disparity,
    const int *d_pixel_ind_offset) {
  int n_val_accum = gridDim.x * blockDim.x; // _MAX_N_VAL_ACCUM may not be
                                            // multiple of blocksize

  int n_disparity = d_n_values_disparity[blockIdx.y];
  int n_accum = (int)ceilf((float)n_disparity / (float)n_val_accum);
  int start_ind = d_start_ind_disparity[blockIdx.y];

  float f = d_focal_length[blockIdx.y];
  float ox = d_nodal_point_x[blockIdx.y];
  float oy = d_nodal_point_y[blockIdx.y];
  float b = d_baseline[blockIdx.y];
  int n_cols = d_n_cols[blockIdx.y];
  int pixel_ind_offset = d_pixel_ind_offset[blockIdx.y];

  // initialize accumulators

  float A0 = 0.0f, A1 = 0.0f, A2 = 0.0f, A3 = 0.0f, A4 = 0.0f, A5 = 0.0f,
        A6 = 0.0f, A7 = 0.0f, A8 = 0.0f, A9 = 0.0f, A10 = 0.0f, A11 = 0.0f,
        A12 = 0.0f, A13 = 0.0f, A14 = 0.0f, A15 = 0.0f, A16 = 0.0f, A17 = 0.0f,
        A18 = 0.0f, A19 = 0.0f, A20 = 0.0f, A21 = 0.0f, A22 = 0.0f, A23 = 0.0f,
        A24 = 0.0f, A25 = 0.0f, A26 = 0.0f;

  for (int in_ind = blockDim.x * blockIdx.x * n_accum + threadIdx.x;
       in_ind < blockDim.x * (blockIdx.x + 1) * n_accum; in_ind += blockDim.x) {

    if (in_ind < n_disparity) { // is this a valid sample?

      // fetch disparity, Zbuffer and normal from global memory
      float disp = d_disparity_compact[in_ind + start_ind];
      float4 tmp = d_Zbuffer_normals_compact[in_ind + start_ind];
      float Zbuffer = tmp.x;
      float nx = tmp.y;
      float ny = tmp.z;
      float nz = tmp.w;

      // compute coordinates
      int pixel_ind =
          d_ind_disparity_Zbuffer[in_ind + start_ind] - pixel_ind_offset;

      float y = floorf(__fdividef((float)pixel_ind, n_cols));
      float x = (float)pixel_ind - y * n_cols;

      x = __fdividef((x - ox), f);
      y = -__fdividef((y - oy), f);

      // reconstruct 3D point from disparity

      float Zd = -(f * b) / disp;
      float Xd = x * Zd;
      float Yd = y * Zd;

      // reconstruct 3D point from model

      float Zm = Zbuffer;
      float Xm = x * Zm;
      float Ym = y * Zm;

      /************************/
      /* evaluate constraints */
      /************************/

      // unique values A-matrix

      A0 += nx * nx;
      A1 += nx * ny;
      A2 += nx * nz;
      A3 += Ym * nx * nz - Zm * nx * ny;
      A4 += Zm * (nx * nx) - Xm * nx * nz;
      A5 += -Ym * (nx * nx) + Xm * nx * ny;

      A6 += ny * ny;
      A7 += ny * nz;
      A8 += -Zm * (ny * ny) + Ym * ny * nz;
      A9 += -Xm * ny * nz + Zm * nx * ny;
      A10 += Xm * (ny * ny) - Ym * nx * ny;

      A11 += nz * nz;
      A12 += Ym * (nz * nz) - Zm * ny * nz;
      A13 += -Xm * (nz * nz) + Zm * nx * nz;
      A14 += Xm * ny * nz - Ym * nx * nz;

      A15 += (Ym * Ym) * (nz * nz) + (Zm * Zm) * (ny * ny) -
             Ym * Zm * ny * nz * 2.0f;
      A16 += -Xm * Ym * (nz * nz) - (Zm * Zm) * nx * ny + Xm * Zm * ny * nz +
             Ym * Zm * nx * nz;
      A17 += -Xm * Zm * (ny * ny) - (Ym * Ym) * nx * nz + Xm * Ym * ny * nz +
             Ym * Zm * nx * ny;

      A18 += (Xm * Xm) * (nz * nz) + (Zm * Zm) * (nx * nx) -
             Xm * Zm * nx * nz * 2.0f;
      A19 += -Ym * Zm * (nx * nx) - (Xm * Xm) * ny * nz + Xm * Ym * nx * nz +
             Xm * Zm * nx * ny;

      A20 += (Xm * Xm) * (ny * ny) + (Ym * Ym) * (nx * nx) -
             Xm * Ym * nx * ny * 2.0f;

      // B-vector

      A21 += Xd * (nx * nx) - Xm * (nx * nx) + Yd * nx * ny - Ym * nx * ny +
             Zd * nx * nz - Zm * nx * nz;
      A22 += Yd * (ny * ny) - Ym * (ny * ny) + Xd * nx * ny - Xm * nx * ny +
             Zd * ny * nz - Zm * ny * nz;
      A23 += Zd * (nz * nz) - Zm * (nz * nz) + Xd * nx * nz - Xm * nx * nz +
             Yd * ny * nz - Ym * ny * nz;
      A24 += -Yd * Zm * (ny * ny) + Ym * Zd * (nz * nz) + Ym * Zm * (ny * ny) -
             Ym * Zm * (nz * nz) - (Ym * Ym) * ny * nz + (Zm * Zm) * ny * nz +
             Xd * Ym * nx * nz - Xm * Ym * nx * nz - Xd * Zm * nx * ny +
             Yd * Ym * ny * nz + Xm * Zm * nx * ny - Zd * Zm * ny * nz;
      A25 += Xd * Zm * (nx * nx) - Xm * Zd * (nz * nz) - Xm * Zm * (nx * nx) +
             Xm * Zm * (nz * nz) + (Xm * Xm) * nx * nz - (Zm * Zm) * nx * nz -
             Xd * Xm * nx * nz - Xm * Yd * ny * nz + Xm * Ym * ny * nz +
             Yd * Zm * nx * ny - Ym * Zm * nx * ny + Zd * Zm * nx * nz;
      A26 += -Xd * Ym * (nx * nx) + Xm * Yd * (ny * ny) + Xm * Ym * (nx * nx) -
             Xm * Ym * (ny * ny) - (Xm * Xm) * nx * ny + (Ym * Ym) * nx * ny +
             Xd * Xm * nx * ny - Yd * Ym * nx * ny + Xm * Zd * ny * nz -
             Xm * Zm * ny * nz - Ym * Zd * nx * nz + Ym * Zm * nx * nz;
    }
  }

  /**************************/
  /* write out accumulators */
  /**************************/

  int out_ind =
      27 * n_val_accum * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

  d_CD[out_ind] = A0;
  d_CD[out_ind + n_val_accum] = A1;
  d_CD[out_ind + 2 * n_val_accum] = A2;
  d_CD[out_ind + 3 * n_val_accum] = A3;
  d_CD[out_ind + 4 * n_val_accum] = A4;
  d_CD[out_ind + 5 * n_val_accum] = A5;
  d_CD[out_ind + 6 * n_val_accum] = A6;
  d_CD[out_ind + 7 * n_val_accum] = A7;
  d_CD[out_ind + 8 * n_val_accum] = A8;
  d_CD[out_ind + 9 * n_val_accum] = A9;
  d_CD[out_ind + 10 * n_val_accum] = A10;
  d_CD[out_ind + 11 * n_val_accum] = A11;
  d_CD[out_ind + 12 * n_val_accum] = A12;
  d_CD[out_ind + 13 * n_val_accum] = A13;
  d_CD[out_ind + 14 * n_val_accum] = A14;
  d_CD[out_ind + 15 * n_val_accum] = A15;
  d_CD[out_ind + 16 * n_val_accum] = A16;
  d_CD[out_ind + 17 * n_val_accum] = A17;
  d_CD[out_ind + 18 * n_val_accum] = A18;
  d_CD[out_ind + 19 * n_val_accum] = A19;
  d_CD[out_ind + 20 * n_val_accum] = A20;
  d_CD[out_ind + 21 * n_val_accum] = A21;
  d_CD[out_ind + 22 * n_val_accum] = A22;
  d_CD[out_ind + 23 * n_val_accum] = A23;
  d_CD[out_ind + 24 * n_val_accum] = A24;
  d_CD[out_ind + 25 * n_val_accum] = A25;
  d_CD[out_ind + 26 * n_val_accum] = A26;
}

// Final reduction of the normal equations

__global__ void reduce_normal_eqs_64_GPU(float *d_C_reduced, float *d_C,
                                         int gridDim_x_normal_equations) {

  int tid = threadIdx.x;
  int bx = blockIdx.x;
  // put data in shared memory

  int ind = blockIdx.y * gridDim.x * gridDim_x_normal_equations * 64 +
            bx * gridDim_x_normal_equations * 64 + tid;

  __shared__ float DATA[64];

  // load and sum the first 20 elements
  float tmp = 0.0f;
  for (int i = 0; i < gridDim_x_normal_equations; i++)
    tmp += d_C[ind + i * 64];
  DATA[tid] = tmp;

  __syncthreads(); // ensure reading stage has finished

  // reduction
  if (tid < 32) { // warp-reduce
    DATA[tid] += DATA[tid + 32];
    __syncthreads();
    DATA[tid] += DATA[tid + 16];
    __syncthreads();
    DATA[tid] += DATA[tid + 8];
    __syncthreads();
    DATA[tid] += DATA[tid + 4];
    __syncthreads();
    DATA[tid] += DATA[tid + 2];
    __syncthreads();
    DATA[tid] += DATA[tid + 1];
    __syncthreads();
  }

  // write results
  if (tid == 0)
    d_C_reduced[blockIdx.y * gridDim.x + bx] = DATA[0];
}

// Final reduction of the normal equations
// In this version each block processes multiple constraints (according to
// threadIdx.y)

__global__ void
reduce_normal_eqs_64_mult_constr_GPU(float *d_C_reduced, const float *d_C,
                                     int gridDim_x_normal_equations,
                                     int n_constraints) {
  // check if there are constraints left to be processed
  int constraint_ind = blockIdx.x * 4 + threadIdx.y;

  if (constraint_ind < n_constraints) {

    int tid = 64 * threadIdx.y + threadIdx.x;

    // put data in shared memory
    int ind = blockIdx.y * n_constraints * gridDim_x_normal_equations * 64 +
              constraint_ind * gridDim_x_normal_equations * 64 + threadIdx.x;

    __shared__ float DATA[64 * 4];

    // load and sum the first gridDim_x_normal_equations elements
    float tmp = 0.0f;
    for (int i = 0; i < gridDim_x_normal_equations; i++)
      tmp += d_C[ind + i * 64];
    DATA[tid] = tmp;

    __syncthreads(); // ensure reading stage has finished

    if ((tid - 64 * threadIdx.y) < 32) { // warp-reduce
      DATA[tid] += DATA[tid + 32];
      __syncthreads();
      DATA[tid] += DATA[tid + 16];
      __syncthreads();
      DATA[tid] += DATA[tid + 8];
      __syncthreads();
      DATA[tid] += DATA[tid + 4];
      __syncthreads();
      DATA[tid] += DATA[tid + 2];
      __syncthreads();
      DATA[tid] += DATA[tid + 1];
      __syncthreads();
    }

    // write results
    if (threadIdx.x == 0)
      d_C_reduced[blockIdx.y * n_constraints + constraint_ind] = DATA[tid];
  }
}

// Auxiliary device functions to compute OLS absolute residual

__device__ static float flow_absolute_residual(float x, float y, float ux,
                                               float uy, float d, float fx,
                                               float fy, float T0, float T1,
                                               float T2, float R0, float R1,
                                               float R2) {
  float rx = -ux + fx * R1 - y * R2 + ((x * x) * R1) / fx + d * fx * T0 -
             d * x * T2 - (x * y * R0) / fx;
  float ry = -uy - fy * R0 + x * R2 - d * y * T2 - ((y * y) * R0) / fy +
             d * fy * T1 + (x * y * R1) / fy;

  return sqrtf(rx * rx + ry * ry);
}

__device__ static float disp_absolute_residual(float Xd, float Yd, float Zd,
                                               float Xm, float Ym, float Zm,
                                               float nx, float ny, float nz,
                                               float T0, float T1, float T2,
                                               float R0, float R1, float R2,
                                               float fx, float b) {
  float r = -Xd * nx + Xm * nx - Yd * ny + Ym * ny - Zd * nz + Zm * nz +
            nx * T0 + ny * T1 + nz * T2 + Xm * ny * R2 - Xm * nz * R1 -
            Ym * nx * R2 + Ym * nz * R0 + Zm * nx * R1 - Zm * ny * R0;

  // weight to convert distance units to pixels
  r *= fx * b / (Zm * Zm);

  return fabsf(r);
}

// Absolute residual for flow multi-camera case

//__global__ void flow_absolute_residual_multicam_GPU(float *d_abs_res, float2
//*d_flow_compact, float *d_Zbuffer_flow_compact, int *d_ind_flow_Zbuffer,
//unsigned int *d_valid_flow_Zbuffer, const float* d_focal_length, const float*
//d_nodal_point_x, const float* d_nodal_point_y, const int* d_n_rows, const int*
//d_n_cols, int n_valid_flow_Zbuffer, const int *d_res_offset_ind, const int
//*d_pixel_ind_offset, const int *d_segment_translation_table, const float
//*d_dTR)
//{

//  int ind = blockDim.x*blockIdx.x + threadIdx.x;

//  if (ind < n_valid_flow_Zbuffer) {

//    // determine current segment
//    int segment = d_segment_translation_table[d_valid_flow_Zbuffer[ind]];

//    // get segment parameters
//    float f = d_focal_length[segment];
//    float ox = d_nodal_point_x[segment];
//    float oy = d_nodal_point_y[segment];
//    int n_rows = d_n_rows[segment];
//    int n_cols = d_n_cols[segment];
//    int pixel_ind_offset = d_pixel_ind_offset[segment];
//    int res_offset = d_res_offset_ind[segment];

//    // fetch flow and Zbuffer from global memory
//    float2 u = d_flow_compact[ind];
//    float disp = __fdividef(1.0f,d_Zbuffer_flow_compact[ind]);

//    // compute coordinates
//    int pixel_ind = d_ind_flow_Zbuffer[ind] - pixel_ind_offset;
//    bool is_ar_flow = (pixel_ind>=(n_rows*n_cols));
//    pixel_ind -= (int)is_ar_flow*n_rows*n_cols;

//    float y = floorf(__fdividef( (float)pixel_ind , n_cols ));
//    float x = (float)pixel_ind - y*n_cols;

//    x = x - ox;
//    y = y - oy;

//    // flip y axis
//    y = -y;
//    u.y = -u.y;

//    // compute absolute residual
//    // here the weights will be introduced
//    int ind_out = ind + res_offset;
//    int s6 = segment*6;
//    d_abs_res[ind_out] = flow_absolute_residual(x, y, u.x, u.y, disp, f,
// d_dTR[s6], d_dTR[s6+1], d_dTR[s6+2], d_dTR[s6+3], d_dTR[s6+4], d_dTR[s6+5]);

//  }

//}

// Absolute residual for flow

__global__ void flow_absolute_residual_scalable_GPU(
    float *d_abs_res, const float2 *d_flow_compact,
    const float *d_Zbuffer_flow_compact, const int *d_ind_flow_Zbuffer,
    const unsigned int *d_valid_flow_Zbuffer, float fx, float fy, float ox,
    float oy, int n_rows, int n_cols, int n_valid_flow_Zbuffer,
    const int *d_offset_ind, const int *d_segment_translation_table,
    float w_flow, float w_ar_flow, const float *d_dTR) {

  int ind = blockDim.x * blockIdx.x + threadIdx.x;

  if (ind < n_valid_flow_Zbuffer) {

    // determine current segment
    int segment = d_segment_translation_table[d_valid_flow_Zbuffer[ind]];

    // fetch flow and Zbuffer from global memory
    float2 u = d_flow_compact[ind];
    float disp = __fdividef(1.0f, d_Zbuffer_flow_compact[ind]);

    // compute coordinates
    int pixel_ind = d_ind_flow_Zbuffer[ind];
    bool is_ar_flow = (pixel_ind >= (n_rows * n_cols));
    pixel_ind -= (int)is_ar_flow * n_rows * n_cols;

    float y = floorf(__fdividef((float)pixel_ind, n_cols));
    float x = (float)pixel_ind - y * n_cols;

    x = x - ox;
    y = y - oy;

    // compute absolute residual
    // here the weights will be introduced
    float w = is_ar_flow ? w_ar_flow : w_flow;
    int ind_out = ind + d_offset_ind[segment];
    int s6 = segment * 6;
    d_abs_res[ind_out] =
        w * flow_absolute_residual(x, y, u.x, u.y, disp, fx, fy, d_dTR[s6],
                                   d_dTR[s6 + 1], d_dTR[s6 + 2], d_dTR[s6 + 3],
                                   d_dTR[s6 + 4], d_dTR[s6 + 5]);
  }
}

// Absolute residual for disparity multi-camera case

//__global__ void disp_absolute_residual_multicam_GPU(float *d_abs_res, float
//*d_disparity_compact, float4 *d_Zbuffer_normals_compact, int
//*d_ind_disparity_Zbuffer, unsigned int *d_valid_disparity_Zbuffer, const
//float* d_focal_length, const float* d_nodal_point_x, const float*
//d_nodal_point_y, const float * d_baseline, const int* d_n_cols, int
//n_valid_disparity_Zbuffer, const int *d_res_offset_ind, const int
//*d_pixel_ind_offset, const int *d_segment_translation_table, const float
//*d_dTR)
//{

//  int ind = blockDim.x*blockIdx.x + threadIdx.x;

//  if (ind < n_valid_disparity_Zbuffer) {

//    // determine current segment
//    int segment = d_segment_translation_table[d_valid_disparity_Zbuffer[ind]];

//    // get segment parameters
//    float f = d_focal_length[segment];
//    float ox = d_nodal_point_x[segment];
//    float oy = d_nodal_point_y[segment];
//    float b = d_baseline[segment];
//    int n_cols = d_n_cols[segment];
//    int pixel_ind_offset = d_pixel_ind_offset[segment];
//    int res_offset = d_res_offset_ind[segment];

//    // fetch disparity, Zbuffer and normal from global memory
//    float disp = d_disparity_compact[ind];
//    float4 tmp = d_Zbuffer_normals_compact[ind];
//    float Zbuffer = tmp.x;
//    float nx = tmp.y;
//    float ny = tmp.z;
//    float nz = tmp.w;

//    // compute coordinates
//    int pixel_ind = d_ind_disparity_Zbuffer[ind] - pixel_ind_offset;

//    float y = floorf(__fdividef( (float)pixel_ind , n_cols ));
//    float x = (float)pixel_ind - y*n_cols;

//    x = __fdividef( (x - ox) , f );
//    y = -__fdividef( (y - oy) , f );

//    // reconstruct 3D point from disparity
//    float Zd = -(f*b)/disp;
//    float Xd = x*Zd;
//    float Yd = y*Zd;

//    // reconstruct 3D point from model
//    float Zm = Zbuffer;
//    float Xm = x*Zm;
//    float Ym = y*Zm;

//    // compute absolute residual (weighted by disparity vs flow importance)
//    int ind_out = ind + res_offset;
//    int s6 = segment*6;
//    d_abs_res[ind_out] = disp_absolute_residual(Xd, Yd, Zd, Xm, Ym, Zm, nx,
// ny, nz, d_dTR[s6], d_dTR[s6+1], d_dTR[s6+2], d_dTR[s6+3], d_dTR[s6+4],
// d_dTR[s6+5]);

//  }
//}

// Absolute residual for disparity

__global__ void disp_absolute_residual_scalable_GPU(
    float *d_abs_res, const float *d_disparity_compact,
    const float4 *d_Zbuffer_normals_compact, const int *d_ind_disparity_Zbuffer,
    const unsigned int *d_valid_disparity_Zbuffer, float fx, float fy, float ox,
    float oy, float b, int n_cols, int n_valid_disparity_Zbuffer,
    const int *d_offset_ind, const int *d_segment_translation_table,
    float w_disp, const float *d_dTR) {

  int ind = blockDim.x * blockIdx.x + threadIdx.x;

  if (ind < n_valid_disparity_Zbuffer) {

    // determine current segment
    int segment = d_segment_translation_table[d_valid_disparity_Zbuffer[ind]];

    // fetch disparity, Zbuffer and normal from global memory
    float disp = d_disparity_compact[ind];
    float4 tmp = d_Zbuffer_normals_compact[ind];
    float Zbuffer = tmp.x;
    float nx = tmp.y;
    float ny = tmp.z;
    float nz = tmp.w;

    // compute coordinates
    int pixel_ind = d_ind_disparity_Zbuffer[ind];

    float y = floorf(__fdividef((float)pixel_ind, n_cols));
    float x = (float)pixel_ind - y * n_cols;

    x = __fdividef((x - ox), fx);
    y = __fdividef((y - oy), fy);

    // reconstruct 3D point from disparity
    float Zd = -(fx * b) / disp; // arbitrary use of fx for now
    float Xd = x * Zd;
    float Yd = y * Zd;

    // reconstruct 3D point from model
    float Zm = Zbuffer;
    float Xm = x * Zm;
    float Ym = y * Zm;

    // compute absolute residual (weighted by disparity vs flow importance)
    int ind_out = ind + d_offset_ind[segment];
    int s6 = segment * 6;
    d_abs_res[ind_out] =
        w_disp * disp_absolute_residual(Xd, Yd, Zd, Xm, Ym, Zm, nx, ny, nz,
                                        d_dTR[s6], d_dTR[s6 + 1], d_dTR[s6 + 2],
                                        d_dTR[s6 + 3], d_dTR[s6 + 4],
                                        d_dTR[s6 + 5], fx, b);
  }
}

// Weighted normal equations for flow - multicam case

//__global__ void normal_eqs_flow_weighted_multicam_GPU(float *d_CO, float2
//*d_flow_compact, float *d_Zbuffer_flow_compact, int *d_ind_flow_Zbuffer, const
//float* d_focal_length, const float* d_nodal_point_x, const float*
//d_nodal_point_y, const int* d_n_rows, const int* d_n_cols, const int
//*d_n_values_flow, const int *d_start_ind_flow, const int *d_pixel_ind_offset,
//const float *d_abs_res_scales, const float *d_dTR)
//{

//  int n_val_accum = gridDim.x*blockDim.x; // _MAX_N_VAL_ACCUM may not be
// multiple of blocksize

//  int n_flow = d_n_values_flow[blockIdx.y];
//  int n_accum = (int)ceilf((float)n_flow / (float)n_val_accum);
//  int start_ind = d_start_ind_flow[blockIdx.y];

//  float f = d_focal_length[blockIdx.y];
//  float ox = d_nodal_point_x[blockIdx.y];
//  float oy = d_nodal_point_y[blockIdx.y];
//  int n_rows = d_n_rows[blockIdx.y];
//  int n_cols = d_n_cols[blockIdx.y];
//  int pixel_ind_offset = d_pixel_ind_offset[blockIdx.y];

//  // initialize accumulators

//  float A0 = 0.0f, A1 = 0.0f, A2 = 0.0f, A3 = 0.0f, A4 = 0.0f, A5 = 0.0f, A6 =
// 0.0f, A7 = 0.0f, A8 = 0.0f, A9 = 0.0f, A10 = 0.0f, A11 = 0.0f, A12 = 0.0f,
// A13 = 0.0f, A14 = 0.0f, A15 = 0.0f, A16 = 0.0f, A17 = 0.0f, A18 = 0.0f, A19 =
// 0.0f, A20 = 0.0f, A21 = 0.0f, A22 = 0.0f;

//  for (int in_ind = blockDim.x*blockIdx.x*n_accum + threadIdx.x ; in_ind <
// blockDim.x*(blockIdx.x+1)*n_accum ; in_ind += blockDim.x) {

//    if (in_ind < n_flow ) { // is this a valid sample?

//      // fetch flow and Zbuffer from global memory
//      float2 u = d_flow_compact[in_ind+start_ind];
//      float disp = __fdividef(1.0f,d_Zbuffer_flow_compact[in_ind+start_ind]);

//      // compute coordinates
//      int pixel_ind = d_ind_flow_Zbuffer[in_ind+start_ind] - pixel_ind_offset;
//      bool is_ar_flow = (pixel_ind>=(n_rows*n_cols));
//      pixel_ind -= (int)is_ar_flow*n_rows*n_cols;

//      float y = floorf(__fdividef( (float)pixel_ind , n_cols ));
//      float x = (float)pixel_ind - y*n_cols;

//      x = x - ox;
//      y = y - oy;

//      // flip y axis
//      y = -y;
//      u.y = -u.y;

//      // determine M-estimation weight
////      float w_rel = is_ar_flow ? w_ar_flow : w_flow;
//      int s6 = blockIdx.y*6;
//      float w = flow_absolute_residual(x, y, u.x, u.y, disp, f, d_dTR[s6],
// d_dTR[s6+1], d_dTR[s6+2], d_dTR[s6+3], d_dTR[s6+4], d_dTR[s6+5]);
//      w /= d_abs_res_scales[blockIdx.y];
//      w = (w>1) ? 0 : (1.0f-2.0f*w*w + w*w*w*w);

//      /************************/
//      /* evaluate constraints */
//      /************************/

//      // unique values A-matrix

//      A0 += w * (disp*disp*f*f);
//      A1 += w * (-disp*disp*x*f);
//      A2 += w * (-disp*x*y);
//      A3 += w * (disp*f*f + disp*x*x);
//      A4 += w * (-disp*y*f);
//      A5 += w * (-disp*disp*y*f);
//      A6 += w * (-disp*f*f - disp*y*y);
//      A7 += w * (disp*x*f);
//      A8 += w * (disp*disp*x*x + disp*disp*y*y);
//      A9 += w * (disp*x*x*y/f + disp*y*f + disp*y*y*y/f);
//      A10 += w * (-disp*x*f - disp*x*x*x/f - disp*x*y*y/f);
//      A11 += w * (x*x*y*y/(f*f) + f*f + 2.0f*y*y + y*y*y*y/(f*f));
//      A12 += w * (-2.0f*x*y - x*x*x*y/(f*f) - x*y*y*y/(f*f));
//      A13 += w * (-x*f);
//      A14 += w * (f*f + 2.0f*x*x + x*x*x*x/(f*f) + x*x*y*y/(f*f));
//      A15 += w * (-y*f);
//      A16 += w * (x*x + y*y);

//      // B-vector

//      A17 += w * (disp*u.x*f);
//      A18 += w * (disp*u.y*f);
//      A19 += w * (-disp*x*u.x - disp*y*u.y);
//      A20 += w * (-x*y*u.x/f - u.y*f - u.y*y*y/f);
//      A21 += w * (u.x*f + x*x*u.x/f + x*y*u.y/f);
//      A22 += w * (-y*u.x + x*u.y);

//    }

//  }

//  /**************************/
//  /* write out accumulators */
//  /**************************/

//  int out_ind = 23*n_val_accum*blockIdx.y + blockDim.x*blockIdx.x +
// threadIdx.x;

//  d_CO[out_ind] = A0;
//  d_CO[out_ind+n_val_accum] = A1;
//  d_CO[out_ind+2*n_val_accum] = A2;
//  d_CO[out_ind+3*n_val_accum] = A3;
//  d_CO[out_ind+4*n_val_accum] = A4;
//  d_CO[out_ind+5*n_val_accum] = A5;
//  d_CO[out_ind+6*n_val_accum] = A6;
//  d_CO[out_ind+7*n_val_accum] = A7;
//  d_CO[out_ind+8*n_val_accum] = A8;
//  d_CO[out_ind+9*n_val_accum] = A9;
//  d_CO[out_ind+10*n_val_accum] = A10;
//  d_CO[out_ind+11*n_val_accum] = A11;
//  d_CO[out_ind+12*n_val_accum] = A12;
//  d_CO[out_ind+13*n_val_accum] = A13;
//  d_CO[out_ind+14*n_val_accum] = A14;
//  d_CO[out_ind+15*n_val_accum] = A15;
//  d_CO[out_ind+16*n_val_accum] = A16;
//  d_CO[out_ind+17*n_val_accum] = A17;
//  d_CO[out_ind+18*n_val_accum] = A18;
//  d_CO[out_ind+19*n_val_accum] = A19;
//  d_CO[out_ind+20*n_val_accum] = A20;
//  d_CO[out_ind+21*n_val_accum] = A21;
//  d_CO[out_ind+22*n_val_accum] = A22;

//}

// Weighted normal equations for flow

__global__ void normal_eqs_flow_weighted_GPU(
    float *d_CO, const float2 *d_flow_compact,
    const float *d_Zbuffer_flow_compact, const int *d_ind_flow_Zbuffer,
    float fx, float fy, float ox, float oy, int n_rows, int n_cols,
    const int *d_n_values_flow, const int *d_start_ind_flow,
    const float *d_abs_res_scales, float w_flow, float w_ar_flow,
    const float *d_dTR) {

  int n_val_accum = gridDim.x * blockDim.x; // _MAX_N_VAL_ACCUM may not be
                                            // multiple of blocksize

  int n_flow = d_n_values_flow[blockIdx.y];
  int n_accum = (int)ceilf((float)n_flow / (float)n_val_accum);
  int start_ind = d_start_ind_flow[blockIdx.y];

  // initialize accumulators

  float A0 = 0.0f, A1 = 0.0f, A2 = 0.0f, A3 = 0.0f, A4 = 0.0f, A5 = 0.0f,
        A6 = 0.0f, A7 = 0.0f, A8 = 0.0f, A9 = 0.0f, A10 = 0.0f, A11 = 0.0f,
        A12 = 0.0f, A13 = 0.0f, A14 = 0.0f, A15 = 0.0f, A16 = 0.0f, A17 = 0.0f,
        A18 = 0.0f, A19 = 0.0f, A20 = 0.0f, A21 = 0.0f, A22 = 0.0f;

  for (int in_ind = blockDim.x * blockIdx.x * n_accum + threadIdx.x;
       in_ind < blockDim.x * (blockIdx.x + 1) * n_accum; in_ind += blockDim.x) {

    if (in_ind < n_flow) { // is this a valid sample?

      // fetch flow and Zbuffer from global memory
      float2 u = d_flow_compact[in_ind + start_ind];
      float disp = __fdividef(1.0f, d_Zbuffer_flow_compact[in_ind + start_ind]);

      // compute coordinates
      int pixel_ind = d_ind_flow_Zbuffer[in_ind + start_ind];
      bool is_ar_flow = (pixel_ind >= (n_rows * n_cols));
      pixel_ind -= (int)is_ar_flow * n_rows * n_cols;

      float y = floorf(__fdividef((float)pixel_ind, n_cols));
      float x = (float)pixel_ind - y * n_cols;

      x = x - ox;
      y = y - oy;

      // determine M-estimation weight
      float w_rel = is_ar_flow ? w_ar_flow : w_flow;
      int s6 = blockIdx.y * 6;
      float w = w_rel * flow_absolute_residual(x, y, u.x, u.y, disp, fx, fy,
                                               d_dTR[s6], d_dTR[s6 + 1],
                                               d_dTR[s6 + 2], d_dTR[s6 + 3],
                                               d_dTR[s6 + 4], d_dTR[s6 + 5]);
      w /= d_abs_res_scales[blockIdx.y];
      w = (w > 1) ? 0 : (1.0f - 2.0f * w * w + w * w * w * w);

      /************************/
      /* evaluate constraints */
      /************************/

      // unique values A-matrix

      A0 += w * (disp * disp * fx * fx);
      A1 += w * (-disp * disp * x * fx);
      A2 += w * (-disp * x * y);
      A3 += w * (disp * fx * fx + disp * x * x);
      A4 += w * (-disp * y * fx);
      A5 += w * (-disp * disp * y * fy);
      A6 += w * (-disp * fy * fy - disp * y * y); //!!!!
      A7 += w * (disp * x * fy);
      A8 += w * (disp * disp * x * x + disp * disp * y * y);
      A9 += w * (disp * x * x * y / fx + disp * y * fy + disp * y * y * y / fy);
      A10 +=
          w * (-disp * x * fx - disp * x * x * x / fx - disp * x * y * y / fy);
      A11 += w * (x * x * y * y / (fx * fx) + fy * fy + 2.0f * y * y +
                  y * y * y * y / (fy * fy));
      A12 += w * (-2.0f * x * y - x * x * x * y / (fx * fx) -
                  x * y * y * y / (fy * fy));
      A13 += w * (x * y * y / fx - x * fy - x * y * y / fy);
      A14 += w * (fx * fx + 2.0f * x * x + x * x * x * x / (fx * fx) +
                  x * x * y * y / (fy * fy));
      A15 += w * (-y * fx - x * x * y / fx + x * x * y / fy);
      A16 += w * (x * x + y * y);

      // B-vector

      A17 += w * (disp * u.x * fx);
      A18 += w * (disp * u.y * fy);
      A19 += w * (-disp * x * u.x - disp * y * u.y);
      A20 += w * (-x * y * u.x / fx - u.y * fy - u.y * y * y / fy);
      A21 += w * (u.x * fx + x * x * u.x / fx + x * y * u.y / fy);
      A22 += w * (-y * u.x + x * u.y);
    }
  }

  /**************************/
  /* write out accumulators */
  /**************************/

  int out_ind =
      23 * n_val_accum * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

  d_CO[out_ind] = A0;
  d_CO[out_ind + n_val_accum] = A1;
  d_CO[out_ind + 2 * n_val_accum] = A2;
  d_CO[out_ind + 3 * n_val_accum] = A3;
  d_CO[out_ind + 4 * n_val_accum] = A4;
  d_CO[out_ind + 5 * n_val_accum] = A5;
  d_CO[out_ind + 6 * n_val_accum] = A6;
  d_CO[out_ind + 7 * n_val_accum] = A7;
  d_CO[out_ind + 8 * n_val_accum] = A8;
  d_CO[out_ind + 9 * n_val_accum] = A9;
  d_CO[out_ind + 10 * n_val_accum] = A10;
  d_CO[out_ind + 11 * n_val_accum] = A11;
  d_CO[out_ind + 12 * n_val_accum] = A12;
  d_CO[out_ind + 13 * n_val_accum] = A13;
  d_CO[out_ind + 14 * n_val_accum] = A14;
  d_CO[out_ind + 15 * n_val_accum] = A15;
  d_CO[out_ind + 16 * n_val_accum] = A16;
  d_CO[out_ind + 17 * n_val_accum] = A17;
  d_CO[out_ind + 18 * n_val_accum] = A18;
  d_CO[out_ind + 19 * n_val_accum] = A19;
  d_CO[out_ind + 20 * n_val_accum] = A20;
  d_CO[out_ind + 21 * n_val_accum] = A21;
  d_CO[out_ind + 22 * n_val_accum] = A22;
}

// Unweighted normal equations for disparity

//__global__ void normal_eqs_disparity_weighted_multicam_GPU(float *d_CD, float
//*d_disparity_compact, float4 *d_Zbuffer_normals_compact, int
//*d_ind_disparity_Zbuffer, const float* d_focal_length, const float*
//d_nodal_point_x, const float* d_nodal_point_y, const float* d_baseline, const
//int* d_n_cols, const int *d_n_values_disparity, const int
//*d_start_ind_disparity, const int *d_pixel_ind_offset, const float
//*d_abs_res_scales, const float *d_dTR)
//{

//  int n_val_accum = gridDim.x*blockDim.x; // n_val_accum may not be multiple
// of blocksize

//  int n_disparity = d_n_values_disparity[blockIdx.y];
//  int n_accum = (int)ceilf((float)n_disparity / (float)n_val_accum);
//  int start_ind = d_start_ind_disparity[blockIdx.y];

//  float f = d_focal_length[blockIdx.y];
//  float ox = d_nodal_point_x[blockIdx.y];
//  float oy = d_nodal_point_y[blockIdx.y];
//  float b = d_baseline[blockIdx.y];
//  int n_cols = d_n_cols[blockIdx.y];
//  int pixel_ind_offset = d_pixel_ind_offset[blockIdx.y];

//  // initialize accumulators

//  float A0 = 0.0f, A1 = 0.0f, A2 = 0.0f, A3 = 0.0f, A4 = 0.0f, A5 = 0.0f, A6 =
// 0.0f, A7 = 0.0f, A8 = 0.0f, A9 = 0.0f, A10 = 0.0f, A11 = 0.0f, A12 = 0.0f,
// A13 = 0.0f, A14 = 0.0f, A15 = 0.0f, A16 = 0.0f, A17 = 0.0f, A18 = 0.0f, A19 =
// 0.0f, A20 = 0.0f, A21 = 0.0f, A22 = 0.0f, A23 = 0.0f, A24 = 0.0f, A25 = 0.0f,
// A26 = 0.0f;

//  for (int in_ind = blockDim.x*blockIdx.x*n_accum + threadIdx.x ; in_ind <
// blockDim.x*(blockIdx.x+1)*n_accum ; in_ind += blockDim.x) {

//    if (in_ind < n_disparity ) { // is this a valid sample?

//      // fetch disparity, Zbuffer and normal from global memory
//      float disp = d_disparity_compact[in_ind+start_ind];
//      float4 tmp = d_Zbuffer_normals_compact[in_ind+start_ind];
//      float Zbuffer = tmp.x;
//      float nx = tmp.y;
//      float ny = tmp.z;
//      float nz = tmp.w;

//      // compute coordinates
//      int pixel_ind = d_ind_disparity_Zbuffer[in_ind+start_ind] -
// pixel_ind_offset;

//      float y = floorf(__fdividef( (float)pixel_ind , n_cols ));
//      float x = (float)pixel_ind - y*n_cols;

//      x = __fdividef( (x - ox) , f );
//      y = -__fdividef( (y - oy) , f );

//      // reconstruct 3D point from disparity

//      float Zd = -(f*b)/disp;
//      float Xd = x*Zd;
//      float Yd = y*Zd;

//      // reconstruct 3D point from model

//      float Zm = Zbuffer;
//      float Xm = x*Zm;
//      float Ym = y*Zm;

//      // determine M-estimation weight
//      // disparity residual weighed by rel. importance disp vs flow
//      int s6 = blockIdx.y*6;
//      float w = disp_absolute_residual(Xd, Yd, Zd, Xm, Ym, Zm, nx, ny, nz,
// d_dTR[s6], d_dTR[s6+1], d_dTR[s6+2], d_dTR[s6+3], d_dTR[s6+4], d_dTR[s6+5]);
//      w /= d_abs_res_scales[blockIdx.y];
//      w = (w>1) ? 0 : (1.0f-2.0f*w*w + w*w*w*w);

//      /************************/
//      /* evaluate constraints */
//      /************************/

//      // unique values A-matrix

//      A0 += w * (nx*nx);
//      A1 += w * (nx*ny);
//      A2 += w * (nx*nz);
//      A3 += w * (Ym*nx*nz-Zm*nx*ny);
//      A4 += w * (Zm*(nx*nx)-Xm*nx*nz);
//      A5 += w * (-Ym*(nx*nx)+Xm*nx*ny);

//      A6 += w * (ny*ny);
//      A7 += w * (ny*nz);
//      A8 += w * (-Zm*(ny*ny)+Ym*ny*nz);
//      A9 += w * (-Xm*ny*nz+Zm*nx*ny);
//      A10 += w * (Xm*(ny*ny)-Ym*nx*ny);

//      A11 += w * (nz*nz);
//      A12 += w * (Ym*(nz*nz)-Zm*ny*nz);
//      A13 += w * (-Xm*(nz*nz)+Zm*nx*nz);
//      A14 += w * (Xm*ny*nz-Ym*nx*nz);

//      A15 += w * ((Ym*Ym)*(nz*nz)+(Zm*Zm)*(ny*ny)-Ym*Zm*ny*nz*2.0f);
//      A16 += w * (-Xm*Ym*(nz*nz)-(Zm*Zm)*nx*ny+Xm*Zm*ny*nz+Ym*Zm*nx*nz);
//      A17 += w * (-Xm*Zm*(ny*ny)-(Ym*Ym)*nx*nz+Xm*Ym*ny*nz+Ym*Zm*nx*ny);

//      A18 += w * ((Xm*Xm)*(nz*nz)+(Zm*Zm)*(nx*nx)-Xm*Zm*nx*nz*2.0f);
//      A19 += w * (-Ym*Zm*(nx*nx)-(Xm*Xm)*ny*nz+Xm*Ym*nx*nz+Xm*Zm*nx*ny);

//      A20 += w * ((Xm*Xm)*(ny*ny)+(Ym*Ym)*(nx*nx)-Xm*Ym*nx*ny*2.0f);

//      // B-vector

//      A21 += w * (Xd*(nx*nx)-Xm*(nx*nx)+Yd*nx*ny-Ym*nx*ny+Zd*nx*nz-Zm*nx*nz);
//      A22 += w * (Yd*(ny*ny)-Ym*(ny*ny)+Xd*nx*ny-Xm*nx*ny+Zd*ny*nz-Zm*ny*nz);
//      A23 += w * (Zd*(nz*nz)-Zm*(nz*nz)+Xd*nx*nz-Xm*nx*nz+Yd*ny*nz-Ym*ny*nz);
//      A24 += w *
// (-Yd*Zm*(ny*ny)+Ym*Zd*(nz*nz)+Ym*Zm*(ny*ny)-Ym*Zm*(nz*nz)-(Ym*Ym)*ny*nz+(Zm*Zm)*ny*nz+Xd*Ym*nx*nz-Xm*Ym*nx*nz-Xd*Zm*nx*ny+Yd*Ym*ny*nz+Xm*Zm*nx*ny-Zd*Zm*ny*nz);
//      A25 += w *
// (Xd*Zm*(nx*nx)-Xm*Zd*(nz*nz)-Xm*Zm*(nx*nx)+Xm*Zm*(nz*nz)+(Xm*Xm)*nx*nz-(Zm*Zm)*nx*nz-Xd*Xm*nx*nz-Xm*Yd*ny*nz+Xm*Ym*ny*nz+Yd*Zm*nx*ny-Ym*Zm*nx*ny+Zd*Zm*nx*nz);
//      A26 += w *
// (-Xd*Ym*(nx*nx)+Xm*Yd*(ny*ny)+Xm*Ym*(nx*nx)-Xm*Ym*(ny*ny)-(Xm*Xm)*nx*ny+(Ym*Ym)*nx*ny+Xd*Xm*nx*ny-Yd*Ym*nx*ny+Xm*Zd*ny*nz-Xm*Zm*ny*nz-Ym*Zd*nx*nz+Ym*Zm*nx*nz);

//    }

//  }

//  /**************************/
//  /* write out accumulators */
//  /**************************/

//  int out_ind = 27*n_val_accum*blockIdx.y + blockDim.x*blockIdx.x +
// threadIdx.x;

//  d_CD[out_ind] = A0;
//  d_CD[out_ind+n_val_accum] = A1;
//  d_CD[out_ind+2*n_val_accum] = A2;
//  d_CD[out_ind+3*n_val_accum] = A3;
//  d_CD[out_ind+4*n_val_accum] = A4;
//  d_CD[out_ind+5*n_val_accum] = A5;
//  d_CD[out_ind+6*n_val_accum] = A6;
//  d_CD[out_ind+7*n_val_accum] = A7;
//  d_CD[out_ind+8*n_val_accum] = A8;
//  d_CD[out_ind+9*n_val_accum] = A9;
//  d_CD[out_ind+10*n_val_accum] = A10;
//  d_CD[out_ind+11*n_val_accum] = A11;
//  d_CD[out_ind+12*n_val_accum] = A12;
//  d_CD[out_ind+13*n_val_accum] = A13;
//  d_CD[out_ind+14*n_val_accum] = A14;
//  d_CD[out_ind+15*n_val_accum] = A15;
//  d_CD[out_ind+16*n_val_accum] = A16;
//  d_CD[out_ind+17*n_val_accum] = A17;
//  d_CD[out_ind+18*n_val_accum] = A18;
//  d_CD[out_ind+19*n_val_accum] = A19;
//  d_CD[out_ind+20*n_val_accum] = A20;
//  d_CD[out_ind+21*n_val_accum] = A21;
//  d_CD[out_ind+22*n_val_accum] = A22;
//  d_CD[out_ind+23*n_val_accum] = A23;
//  d_CD[out_ind+24*n_val_accum] = A24;
//  d_CD[out_ind+25*n_val_accum] = A25;
//  d_CD[out_ind+26*n_val_accum] = A26;

//}

// Weighted normal equations for disparity

__global__ void normal_eqs_disparity_weighted_GPU(
    float *d_CD, const float *d_disparity_compact,
    const float4 *d_Zbuffer_normals_compact, const int *d_ind_disparity_Zbuffer,
    float fx, float fy, float ox, float oy, float b, int n_cols,
    const int *d_n_values_disparity, const int *d_start_ind_disparity,
    const float *d_abs_res_scales, float w_disp, const float *d_dTR) {

  int n_val_accum =
      gridDim.x * blockDim.x; // n_val_accum may not be multiple of blocksize

  int n_disparity = d_n_values_disparity[blockIdx.y];
  int n_accum = (int)ceilf((float)n_disparity / (float)n_val_accum);
  int start_ind = d_start_ind_disparity[blockIdx.y];

  // initialize accumulators

  float A0 = 0.0f, A1 = 0.0f, A2 = 0.0f, A3 = 0.0f, A4 = 0.0f, A5 = 0.0f,
        A6 = 0.0f, A7 = 0.0f, A8 = 0.0f, A9 = 0.0f, A10 = 0.0f, A11 = 0.0f,
        A12 = 0.0f, A13 = 0.0f, A14 = 0.0f, A15 = 0.0f, A16 = 0.0f, A17 = 0.0f,
        A18 = 0.0f, A19 = 0.0f, A20 = 0.0f, A21 = 0.0f, A22 = 0.0f, A23 = 0.0f,
        A24 = 0.0f, A25 = 0.0f, A26 = 0.0f;

  for (int in_ind = blockDim.x * blockIdx.x * n_accum + threadIdx.x;
       in_ind < blockDim.x * (blockIdx.x + 1) * n_accum; in_ind += blockDim.x) {

    if (in_ind < n_disparity) { // is this a valid sample?

      // fetch disparity, Zbuffer and normal from global memory
      float disp = d_disparity_compact[in_ind + start_ind];
      float4 tmp = d_Zbuffer_normals_compact[in_ind + start_ind];
      float Zbuffer = tmp.x;
      float nx = tmp.y;
      float ny = tmp.z;
      float nz = tmp.w;

      // compute coordinates
      int pixel_ind = d_ind_disparity_Zbuffer[in_ind + start_ind];

      float y = floorf(__fdividef((float)pixel_ind, n_cols));
      float x = (float)pixel_ind - y * n_cols;

      x = __fdividef((x - ox), fx);
      y = __fdividef((y - oy), fy);

      // reconstruct 3D point from disparity

      float Zd = -(fx * b) / disp; // arbitrary use of fx
      float Xd = x * Zd;
      float Yd = y * Zd;

      // reconstruct 3D point from model

      float Zm = Zbuffer;
      float Xm = x * Zm;
      float Ym = y * Zm;

      // determine M-estimation weight
      // disparity residual weighed by rel. importance disp vs flow
      int s6 = blockIdx.y * 6;
      float w = w_disp * disp_absolute_residual(
                             Xd, Yd, Zd, Xm, Ym, Zm, nx, ny, nz, d_dTR[s6],
                             d_dTR[s6 + 1], d_dTR[s6 + 2], d_dTR[s6 + 3],
                             d_dTR[s6 + 4], d_dTR[s6 + 5], fx, b);
      w /= d_abs_res_scales[blockIdx.y];
      w = (w > 1) ? 0 : (1.0f - 2.0f * w * w + w * w * w * w);

      // multiply m estimation weight with distance->pixel conversion weight
      // (squared)
      w *= (fx * fx * b * b) / (Zm * Zm * Zm * Zm);

      /************************/
      /* evaluate constraints */
      /************************/

      // unique values A-matrix

      A0 += w * (nx * nx);
      A1 += w * (nx * ny);
      A2 += w * (nx * nz);
      A3 += w * (Ym * nx * nz - Zm * nx * ny);
      A4 += w * (Zm * (nx * nx) - Xm * nx * nz);
      A5 += w * (-Ym * (nx * nx) + Xm * nx * ny);

      A6 += w * (ny * ny);
      A7 += w * (ny * nz);
      A8 += w * (-Zm * (ny * ny) + Ym * ny * nz);
      A9 += w * (-Xm * ny * nz + Zm * nx * ny);
      A10 += w * (Xm * (ny * ny) - Ym * nx * ny);

      A11 += w * (nz * nz);
      A12 += w * (Ym * (nz * nz) - Zm * ny * nz);
      A13 += w * (-Xm * (nz * nz) + Zm * nx * nz);
      A14 += w * (Xm * ny * nz - Ym * nx * nz);

      A15 += w * ((Ym * Ym) * (nz * nz) + (Zm * Zm) * (ny * ny) -
                  Ym * Zm * ny * nz * 2.0f);
      A16 += w * (-Xm * Ym * (nz * nz) - (Zm * Zm) * nx * ny +
                  Xm * Zm * ny * nz + Ym * Zm * nx * nz);
      A17 += w * (-Xm * Zm * (ny * ny) - (Ym * Ym) * nx * nz +
                  Xm * Ym * ny * nz + Ym * Zm * nx * ny);

      A18 += w * ((Xm * Xm) * (nz * nz) + (Zm * Zm) * (nx * nx) -
                  Xm * Zm * nx * nz * 2.0f);
      A19 += w * (-Ym * Zm * (nx * nx) - (Xm * Xm) * ny * nz +
                  Xm * Ym * nx * nz + Xm * Zm * nx * ny);

      A20 += w * ((Xm * Xm) * (ny * ny) + (Ym * Ym) * (nx * nx) -
                  Xm * Ym * nx * ny * 2.0f);

      // B-vector

      A21 += w * (Xd * (nx * nx) - Xm * (nx * nx) + Yd * nx * ny -
                  Ym * nx * ny + Zd * nx * nz - Zm * nx * nz);
      A22 += w * (Yd * (ny * ny) - Ym * (ny * ny) + Xd * nx * ny -
                  Xm * nx * ny + Zd * ny * nz - Zm * ny * nz);
      A23 += w * (Zd * (nz * nz) - Zm * (nz * nz) + Xd * nx * nz -
                  Xm * nx * nz + Yd * ny * nz - Ym * ny * nz);
      A24 += w *
             (-Yd * Zm * (ny * ny) + Ym * Zd * (nz * nz) + Ym * Zm * (ny * ny) -
              Ym * Zm * (nz * nz) - (Ym * Ym) * ny * nz + (Zm * Zm) * ny * nz +
              Xd * Ym * nx * nz - Xm * Ym * nx * nz - Xd * Zm * nx * ny +
              Yd * Ym * ny * nz + Xm * Zm * nx * ny - Zd * Zm * ny * nz);
      A25 +=
          w * (Xd * Zm * (nx * nx) - Xm * Zd * (nz * nz) - Xm * Zm * (nx * nx) +
               Xm * Zm * (nz * nz) + (Xm * Xm) * nx * nz - (Zm * Zm) * nx * nz -
               Xd * Xm * nx * nz - Xm * Yd * ny * nz + Xm * Ym * ny * nz +
               Yd * Zm * nx * ny - Ym * Zm * nx * ny + Zd * Zm * nx * nz);
      A26 += w *
             (-Xd * Ym * (nx * nx) + Xm * Yd * (ny * ny) + Xm * Ym * (nx * nx) -
              Xm * Ym * (ny * ny) - (Xm * Xm) * nx * ny + (Ym * Ym) * nx * ny +
              Xd * Xm * nx * ny - Yd * Ym * nx * ny + Xm * Zd * ny * nz -
              Xm * Zm * ny * nz - Ym * Zd * nx * nz + Ym * Zm * nx * nz);
    }
  }

  /**************************/
  /* write out accumulators */
  /**************************/

  int out_ind =
      27 * n_val_accum * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

  w_disp *= w_disp; // weight relative to flow

  d_CD[out_ind] = w_disp * A0;
  d_CD[out_ind + n_val_accum] = w_disp * A1;
  d_CD[out_ind + 2 * n_val_accum] = w_disp * A2;
  d_CD[out_ind + 3 * n_val_accum] = w_disp * A3;
  d_CD[out_ind + 4 * n_val_accum] = w_disp * A4;
  d_CD[out_ind + 5 * n_val_accum] = w_disp * A5;
  d_CD[out_ind + 6 * n_val_accum] = w_disp * A6;
  d_CD[out_ind + 7 * n_val_accum] = w_disp * A7;
  d_CD[out_ind + 8 * n_val_accum] = w_disp * A8;
  d_CD[out_ind + 9 * n_val_accum] = w_disp * A9;
  d_CD[out_ind + 10 * n_val_accum] = w_disp * A10;
  d_CD[out_ind + 11 * n_val_accum] = w_disp * A11;
  d_CD[out_ind + 12 * n_val_accum] = w_disp * A12;
  d_CD[out_ind + 13 * n_val_accum] = w_disp * A13;
  d_CD[out_ind + 14 * n_val_accum] = w_disp * A14;
  d_CD[out_ind + 15 * n_val_accum] = w_disp * A15;
  d_CD[out_ind + 16 * n_val_accum] = w_disp * A16;
  d_CD[out_ind + 17 * n_val_accum] = w_disp * A17;
  d_CD[out_ind + 18 * n_val_accum] = w_disp * A18;
  d_CD[out_ind + 19 * n_val_accum] = w_disp * A19;
  d_CD[out_ind + 20 * n_val_accum] = w_disp * A20;
  d_CD[out_ind + 21 * n_val_accum] = w_disp * A21;
  d_CD[out_ind + 22 * n_val_accum] = w_disp * A22;
  d_CD[out_ind + 23 * n_val_accum] = w_disp * A23;
  d_CD[out_ind + 24 * n_val_accum] = w_disp * A24;
  d_CD[out_ind + 25 * n_val_accum] = w_disp * A25;
  d_CD[out_ind + 26 * n_val_accum] = w_disp * A26;
}

///////////////////////
//                   //
//  KERNEL WRAPPERS  //
//                   //
///////////////////////

void computeResidualFlow(float *d_res_flowx, float *d_res_flowy,
                         float *d_res_ar_flowx, float *d_res_ar_flowy,
                         const float *d_flowx, const float *d_flowy,
                         const float *d_ar_flowx, const float *d_ar_flowy,
                         const float *d_delta_T_accum,
                         const float *d_delta_Rmat_accum, const float *d_init_Z,
                         const cudaArray *d_segment_ind, int n_cols, int n_rows,
                         float nodal_point_x, float nodal_point_y,
                         float focal_length_x, float focal_length_y) {
  // Bind textures to arrays
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();
  cudaBindTextureToArray(d_segmentINDArray_texture, d_segment_ind,
                         channelFloat);

  dim3 TB(16, 12, 1);
  dim3 BG(divUp(n_cols, TB.x), divUp(n_rows, TB.y));

  compute_residual_flow_GPU << <BG, TB>>>
      (d_res_flowx, d_res_flowy, d_flowx, d_flowy, d_delta_T_accum,
       d_delta_Rmat_accum, d_init_Z, n_cols, n_rows, nodal_point_x,
       nodal_point_y, focal_length_x, focal_length_y);

  compute_residual_flow_GPU << <BG, TB>>>
      (d_res_ar_flowx, d_res_ar_flowy, d_ar_flowx, d_ar_flowy, d_delta_T_accum,
       d_delta_Rmat_accum, d_init_Z, n_cols, n_rows, nodal_point_x,
       nodal_point_y, focal_length_x, focal_length_y);

  cudaUnbindTexture(d_segmentINDArray_texture);
}

void markValidFlowZbufferAndZbufferZeroBased(
    unsigned int *d_valid_ar_flow_Zbuffer, unsigned int *d_valid_Zbuffer,
    const float *d_ar_flowx, const cudaArray *d_segmentINDArray, int n_cols,
    int n_rows, int n_objects) {
  // Bind textures to arrays
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();
  cudaBindTextureToArray(d_segmentINDArray_texture, d_segmentINDArray,
                         channelFloat);

  // Mark valid locations
  dim3 threadBlock_mark(16, 16, 1);
  dim3 blockGrid_mark(divUp(n_cols, threadBlock_mark.x),
                      divUp(n_rows, threadBlock_mark.y));

  mark_valid_flow_Zbuffer_and_Zbuffer_zero_based_GPU
          << <blockGrid_mark, threadBlock_mark>>>
      (d_valid_ar_flow_Zbuffer, d_valid_Zbuffer, d_ar_flowx, n_cols, n_rows,
       n_objects);

  cudaUnbindTexture(d_segmentINDArray_texture);
}

void mark_with_zero_based_segmentIND(
    unsigned int *d_valid_flow_Zbuffer, unsigned int *d_valid_disparity_Zbuffer,
    const float *d_flowx, const float *d_ar_flowx, const char *d_disparity,
    const cudaArray *d_segmentINDArray, int n_cols, int n_rows, int n_objects,
    int d_disparity_pitch, bool mark_flow, bool mark_ar_flow,
    bool mark_disparity, int segments_to_update) {
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();
  cudaBindTextureToArray(d_segmentINDArray_texture, d_segmentINDArray,
                         channelFloat);

  dim3 threadBlock(16, 16, 1);
  dim3 blockGrid(divUp(n_cols, threadBlock.x), divUp(n_rows, threadBlock.y));

  mark_with_zero_based_segmentIND_GPU << <blockGrid, threadBlock>>>
      (d_valid_flow_Zbuffer, d_valid_disparity_Zbuffer, d_flowx, d_ar_flowx,
       d_disparity, n_cols, n_rows, n_objects, d_disparity_pitch, mark_flow,
       mark_ar_flow, mark_disparity, segments_to_update);

  cudaUnbindTexture(d_segmentINDArray_texture);
}

void subsample_ind_and_labels(int *d_ind_sub, const int *d_ind,
                              unsigned int *d_label_sub,
                              const unsigned int *d_label, int n_out,
                              float inv_sub_factor) {
  dim3 threadBlock(256, 1);
  dim3 blockGrid(divUp(n_out, threadBlock.x), 1);

  subsample_ind_and_labels_GPU << <blockGrid, threadBlock>>>
      (d_ind_sub, d_ind, d_label_sub, d_label, n_out, inv_sub_factor);
}

void gather_valid_flow_Zbuffer(float2 *d_flow_compact, float *d_Zbuffer_compact,
                               const float *d_flowx, const float *d_flowy,
                               const float *d_ar_flowx, const float *d_ar_flowy,
                               int *d_ind_flow_Zbuffer,
                               const cudaArray *d_ZbufferArray,
                               int n_valid_flow_Zbuffer, int n_cols, int n_rows,
                               float Z_conv1, float Z_conv2,
                               int ind_flow_offset) {
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();
  cudaBindTextureToArray(d_Zbuffer_texture, d_ZbufferArray, channelFloat);

  dim3 threadBlock(256, 1);
  dim3 blockGrid(divUp(n_valid_flow_Zbuffer, threadBlock.x), 1);

  gather_valid_flow_Zbuffer_GPU << <blockGrid, threadBlock>>>
      (d_flow_compact, d_Zbuffer_compact, d_flowx, d_flowy, d_ar_flowx,
       d_ar_flowy, d_ind_flow_Zbuffer, n_valid_flow_Zbuffer, n_cols, n_rows,
       Z_conv1, Z_conv2, ind_flow_offset);

  cudaUnbindTexture(d_Zbuffer_texture);
}

void gather_valid_disparity_Zbuffer(
    float *d_disparity_compact, float4 *d_Zbuffer_normals_compact,
    const char *d_disparity, int *d_ind_disparity_Zbuffer,
    const cudaArray *d_ZbufferArray, const cudaArray *d_normalXArray,
    const cudaArray *d_normalYArray, const cudaArray *d_normalZArray,
    int n_valid_disparity_Zbuffer, int n_cols, int n_rows, float Z_conv1,
    float Z_conv2, int disparity_pitch, int ind_disp_offset) {
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();
  cudaBindTextureToArray(d_Zbuffer_texture, d_ZbufferArray, channelFloat);
  cudaBindTextureToArray(d_normalXArray_texture, d_normalXArray, channelFloat);
  cudaBindTextureToArray(d_normalYArray_texture, d_normalYArray, channelFloat);
  cudaBindTextureToArray(d_normalZArray_texture, d_normalZArray, channelFloat);

  dim3 threadBlock(256, 1);
  dim3 blockGrid(divUp(n_valid_disparity_Zbuffer, threadBlock.x), 1);

  gather_valid_disparity_Zbuffer_GPU << <blockGrid, threadBlock>>>
      (d_disparity_compact, d_Zbuffer_normals_compact, d_disparity,
       d_ind_disparity_Zbuffer, n_valid_disparity_Zbuffer, n_cols, n_rows,
       Z_conv1, Z_conv2, disparity_pitch, ind_disp_offset);

  cudaUnbindTexture(d_normalZArray_texture);
  cudaUnbindTexture(d_normalYArray_texture);
  cudaUnbindTexture(d_normalXArray_texture);
  cudaUnbindTexture(d_Zbuffer_texture);
}

void normal_eqs_disparity(dim3 blockGrid, dim3 threadBlock, float *d_CD,
                          const float *d_disparity_compact,
                          const float4 *d_Zbuffer_normals_compact,
                          const int *d_ind_disparity_Zbuffer, float fx,
                          float fy, float ox, float oy, float b, int n_cols,
                          const int *d_n_values_disparity,
                          const int *d_start_ind_disparity, float w_disp) {
  normal_eqs_disparity_GPU << <blockGrid, threadBlock>>>
      (d_CD, d_disparity_compact, d_Zbuffer_normals_compact,
       d_ind_disparity_Zbuffer, fx, fy, ox, oy, b, n_cols, d_n_values_disparity,
       d_start_ind_disparity, w_disp);
}

void reduce_normal_eqs_64_mult_constr(dim3 blockGrid, dim3 threadBlock,
                                      float *d_C_reduced, const float *d_C,
                                      int gridDim_x_normal_equations,
                                      int n_constraints) {
  reduce_normal_eqs_64_mult_constr_GPU << <blockGrid, threadBlock>>>
      (d_C_reduced, d_C, gridDim_x_normal_equations, n_constraints);
}

void flow_absolute_residual_scalable(
    dim3 blockGrid, dim3 threadBlock, float *d_abs_res,
    const float2 *d_flow_compact, const float *d_Zbuffer_flow_compact,
    const int *d_ind_flow_Zbuffer, const unsigned int *d_valid_flow_Zbuffer,
    float fx, float fy, float ox, float oy, int n_rows, int n_cols,
    int n_valid_flow_Zbuffer, const int *d_offset_ind,
    const int *d_segment_translation_table, float w_flow, float w_ar_flow,
    const float *d_dTR) {
  flow_absolute_residual_scalable_GPU << <blockGrid, threadBlock>>>
      (d_abs_res, d_flow_compact, d_Zbuffer_flow_compact, d_ind_flow_Zbuffer,
       d_valid_flow_Zbuffer, fx, fy, ox, oy, n_rows, n_cols,
       n_valid_flow_Zbuffer, d_offset_ind, d_segment_translation_table, w_flow,
       w_ar_flow, d_dTR);
}

void disp_absolute_residual_scalable(
    dim3 blockGrid, dim3 threadBlock, float *d_abs_res,
    const float *d_disparity_compact, const float4 *d_Zbuffer_normals_compact,
    const int *d_ind_disparity_Zbuffer,
    const unsigned int *d_valid_disparity_Zbuffer, float fx, float fy, float ox,
    float oy, float b, int n_cols, int n_valid_disparity_Zbuffer,
    const int *d_offset_ind, const int *d_segment_translation_table,
    float w_disp, const float *d_dTR) {
  disp_absolute_residual_scalable_GPU << <blockGrid, threadBlock>>>
      (d_abs_res, d_disparity_compact, d_Zbuffer_normals_compact,
       d_ind_disparity_Zbuffer, d_valid_disparity_Zbuffer, fx, fy, ox, oy, b,
       n_cols, n_valid_disparity_Zbuffer, d_offset_ind,
       d_segment_translation_table, w_disp, d_dTR);
}

void normal_eqs_flow(dim3 blockGrid, dim3 threadBlock, float *d_CO,
                     const float2 *d_flow_compact,
                     const float *d_Zbuffer_flow_compact,
                     const int *d_ind_flow_Zbuffer, float fx, float fy,
                     float ox, float oy, int n_rows, int n_cols,
                     const int *d_n_values_flow, const int *d_start_ind_flow) {
  // HACK (need to fix): instability arises here when focal lenghts are unequal
  fx = (fx + fy) / 2.0f;
  fy = fx;

  normal_eqs_flow_GPU << <blockGrid, threadBlock>>>
      (d_CO, d_flow_compact, d_Zbuffer_flow_compact, d_ind_flow_Zbuffer, fx, fy,
       ox, oy, n_rows, n_cols, d_n_values_flow, d_start_ind_flow);
}

void normal_eqs_flow_weighted(dim3 blockGrid, dim3 threadBlock, float *d_CO,
                              const float2 *d_flow_compact,
                              const float *d_Zbuffer_flow_compact,
                              const int *d_ind_flow_Zbuffer, float fx, float fy,
                              float ox, float oy, int n_rows, int n_cols,
                              const int *d_n_values_flow,
                              const int *d_start_ind_flow,
                              const float *d_abs_res_scales, float w_flow,
                              float w_ar_flow, const float *d_dTR) {
  // HACK (need to fix): instability arises here when focal lenghts are unequal
  fx = (fx + fy) / 2.0f;
  fy = fx;

  normal_eqs_flow_weighted_GPU << <blockGrid, threadBlock>>>
      (d_CO, d_flow_compact, d_Zbuffer_flow_compact, d_ind_flow_Zbuffer, fx, fy,
       ox, oy, n_rows, n_cols, d_n_values_flow, d_start_ind_flow,
       d_abs_res_scales, w_flow, w_ar_flow, d_dTR);
}

void normal_eqs_disparity_weighted(
    dim3 blockGrid, dim3 threadBlock, float *d_CD,
    const float *d_disparity_compact, const float4 *d_Zbuffer_normals_compact,
    const int *d_ind_disparity_Zbuffer, float fx, float fy, float ox, float oy,
    float b, int n_cols, const int *d_n_values_disparity,
    const int *d_start_ind_disparity, const float *d_abs_res_scales,
    float w_disp, const float *d_dTR) {
  normal_eqs_disparity_weighted_GPU << <blockGrid, threadBlock>>>
      (d_CD, d_disparity_compact, d_Zbuffer_normals_compact,
       d_ind_disparity_Zbuffer, fx, fy, ox, oy, b, n_cols, d_n_values_disparity,
       d_start_ind_disparity, d_abs_res_scales, w_disp, d_dTR);
}
}
