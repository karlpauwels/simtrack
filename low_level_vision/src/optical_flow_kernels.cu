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
#include <utility_kernels.h>
#include <optical_flow_kernels.h>

#define TWO_PI 6.28318530717958623199592694f
#define DC_THR 0.00001f

namespace vision {

// texture references for 2D float Gabor filter outs
// and previous scale optic flow field
texture<float2, 2, cudaReadModeElementType> d_Gabor_texture2;
texture<float2, 2, cudaReadModeElementType> d_o_prev_scale_texture;

// texture reference for consistency check in two-frame optical flow
texture<float2, 2, cudaReadModeElementType> d_frame2_flow_texture;

// IOC masks
__device__ __constant__ float d_FV_X1[] = {
  1.0000000000000000f,  0.9238795325112867f, 0.7071067811865476f,
  0.3826834323650898f,  0.0000000000000001f, -0.3826834323650897f,
  -0.7071067811865475f, -0.9238795325112867f
};
__device__ __constant__ float d_FV_X2[] = {
  0.0000000000000000f, 0.3826834323650898f, 0.7071067811865475f,
  0.9238795325112867f, 1.0000000000000000f, 0.9238795325112867f,
  0.7071067811865476f, 0.3826834323650899f
};
__device__ __constant__ float d_FV_2_X1[] = {
  1.0000000000000000f, 0.8535533905932737f, 0.5000000000000001f,
  0.1464466094067263f, 0.0000000000000000f, 0.1464466094067262f,
  0.4999999999999999f, 0.8535533905932737f
};
__device__ __constant__ float d_FV_2_X2[] = {
  0.0000000000000000f, 0.1464466094067262f, 0.4999999999999999f,
  0.8535533905932737f, 1.0000000000000000f, 0.8535533905932737f,
  0.5000000000000001f, 0.1464466094067263f
};

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

// 3x3 2D median filter ignoring nans (unless more than half the data are nans)

#define s2(a, b)                                                               \
  {                                                                            \
    tmp = a;                                                                   \
    a = fminf(a, b);                                                           \
    b = fmaxf(tmp, b);                                                         \
  }
#define mn3(a, b, c)                                                           \
  s2(a, b);                                                                    \
  s2(a, c);
#define mx3(a, b, c)                                                           \
  s2(b, c);                                                                    \
  s2(a, c);

#define mnmx3(a, b, c)                                                         \
  mx3(a, b, c);                                                                \
  s2(a, b); // 3 exchanges
#define mnmx4(a, b, c, d)                                                      \
  s2(a, b);                                                                    \
  s2(c, d);                                                                    \
  s2(a, c);                                                                    \
  s2(b, d); // 4 exchanges
#define mnmx5(a, b, c, d, e)                                                   \
  s2(a, b);                                                                    \
  s2(c, d);                                                                    \
  mn3(a, c, e);                                                                \
  mx3(b, d, e); // 6 exchanges
#define mnmx6(a, b, c, d, e, f)                                                \
  s2(a, d);                                                                    \
  s2(b, e);                                                                    \
  s2(c, f);                                                                    \
  mn3(a, b, c);                                                                \
  mx3(d, e, f); // 7 exchanges

__global__ void nanmedfilt2_flow_GPU(float2 *d_Image_med, int n_rows,
                                     int n_cols, int pitch, float2 unreliable) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < n_cols) & (y < n_rows)) { // are we in the image?

    float tmp;
    float bigNum = 1000000.0f;
    int valid_count = 0;
    float xt = (float)x + 0.5f;
    float yt = (float)y + 0.5f;

    // pull top six from texture memory

    float2 v[6];

    v[0] = tex2D(d_o_prev_scale_texture, xt - 1.0f, yt - 1.0f);
    if (isfinite(v[0].x)) {
      valid_count++;
    } else {
      bigNum = -bigNum;
      v[0].x = bigNum;
      v[0].y = bigNum;
    }
    v[1] = tex2D(d_o_prev_scale_texture, xt, yt - 1.0f);
    if (isfinite(v[1].x)) {
      valid_count++;
    } else {
      bigNum = -bigNum;
      v[1].x = bigNum;
      v[1].y = bigNum;
    }
    v[2] = tex2D(d_o_prev_scale_texture, xt + 1.0f, yt - 1.0f);
    if (isfinite(v[2].x)) {
      valid_count++;
    } else {
      bigNum = -bigNum;
      v[2].x = bigNum;
      v[2].y = bigNum;
    }
    v[3] = tex2D(d_o_prev_scale_texture, xt - 1.0f, yt);
    if (isfinite(v[3].x)) {
      valid_count++;
    } else {
      bigNum = -bigNum;
      v[3].x = bigNum;
      v[3].y = bigNum;
    }
    v[4] = tex2D(d_o_prev_scale_texture, xt, yt);
    if (isfinite(v[4].x)) {
      valid_count++;
    } else {
      bigNum = -bigNum;
      v[4].x = bigNum;
      v[4].y = bigNum;
    }
    v[5] = tex2D(d_o_prev_scale_texture, xt + 1.0f, yt);
    if (isfinite(v[5].x)) {
      valid_count++;
    } else {
      bigNum = -bigNum;
      v[5].x = bigNum;
      v[5].y = bigNum;
    }

    // with each pass, remove min and max values and add new value
    mnmx6(v[0].x, v[1].x, v[2].x, v[3].x, v[4].x, v[5].x);
    mnmx6(v[0].y, v[1].y, v[2].y, v[3].y, v[4].y, v[5].y);

    v[5] = tex2D(d_o_prev_scale_texture, xt - 1.0f, yt + 1.0f);
    if (isfinite(v[5].x)) {
      valid_count++;
    } else {
      bigNum = -bigNum;
      v[5].x = bigNum;
      v[5].y = bigNum;
    }

    mnmx5(v[1].x, v[2].x, v[3].x, v[4].x, v[5].x);
    mnmx5(v[1].y, v[2].y, v[3].y, v[4].y, v[5].y);

    v[5] = tex2D(d_o_prev_scale_texture, xt, yt + 1.0f);
    if (isfinite(v[5].x)) {
      valid_count++;
    } else {
      bigNum = -bigNum;
      v[5].x = bigNum;
      v[5].y = bigNum;
    }

    mnmx4(v[2].x, v[3].x, v[4].x, v[5].x);
    mnmx4(v[2].y, v[3].y, v[4].y, v[5].y);

    v[5] = tex2D(d_o_prev_scale_texture, xt + 1.0f, yt + 1.0f);
    if (isfinite(v[5].x)) {
      valid_count++;
    } else {
      bigNum = -bigNum;
      v[5].x = bigNum;
      v[5].y = bigNum;
    }

    mnmx3(v[3].x, v[4].x, v[5].x);
    mnmx3(v[3].y, v[4].y, v[5].y);

    // pick the middle one
    *((float2 *)((char *)d_Image_med + y *pitch) + x) =
        (valid_count > 4) ? v[4] : unreliable;
  }
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

// device component velocity function (for loop unrolling)

__device__ static void
component_velocity_two_frames(int o, const float2 *d_Gabor1, int x, int y,
                              float2 op, unsigned int &n_valids,
                              float &sFV_2_X1, float &sFV_2_X2, float &sFV_X1X2,
                              float &sFV_YX1, float &sFV_YX2) {
  bool dc =
      false; // set to true if one of the gabor responses is below dc_thres
  float2 gabor1, gabor2;
  float xt, yt;
  float phase_diff;

  xt = (float)x + op.x + 0.5f;
  yt = (float)y + op.y + 0.5f;

  gabor1 = *d_Gabor1;
  gabor2 = tex2D(d_Gabor_texture2, xt, yt);
  dc = dc | (fabsf(gabor1.x) < DC_THR) | (fabsf(gabor1.y) < DC_THR) |
       (fabsf(gabor2.x) < DC_THR) | (fabsf(gabor2.y) < DC_THR);
  phase_diff = -atan2(gabor2.x * gabor1.y - gabor1.x * gabor2.y,
                      gabor1.x * gabor2.x + gabor1.y * gabor2.y);

  // component velocity
  float beta = phase_diff;

  // update IOC accumulators
  float valid = (float)(!dc);
  n_valids += (int)(valid);
  sFV_2_X1 += d_FV_2_X1[o] * valid;
  sFV_2_X2 += d_FV_2_X2[o] * valid;
  sFV_X1X2 += d_FV_X1[o] * d_FV_X2[o] * valid;
  sFV_YX1 += beta * d_FV_X1[o] * valid;
  sFV_YX2 += beta * d_FV_X2[o] * valid;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

// Core device optical flow computation

__device__ static void comp_optic_flow_two_frames(const char *d_Gabor1,
                                                  int d_Gabor1Pitch, float2 &op,
                                                  int x, int y, int height,
                                                  float unreliable) {
  float Ox, Oy, sFV_YX1, sFV_YX2, sFV_2_X1, sFV_2_X2, sFV_X1X2;
  sFV_YX1 = 0.0f;
  sFV_YX2 = 0.0f;
  sFV_2_X1 = 0.0f;
  sFV_2_X2 = 0.0f;
  sFV_X1X2 = 0.0f;
  unsigned int n_valids = 0;

  /******************************/
  /* Component Velocity and MSE */
  /******************************/

  // unroll the loop

  d_Gabor1 += y * d_Gabor1Pitch + x * sizeof(float2);
  component_velocity_two_frames(0, (const float2 *)d_Gabor1, x, y, op, n_valids,
                                sFV_2_X1, sFV_2_X2, sFV_X1X2, sFV_YX1, sFV_YX2);
  y += height;
  d_Gabor1 += height * d_Gabor1Pitch;
  component_velocity_two_frames(1, (const float2 *)d_Gabor1, x, y, op, n_valids,
                                sFV_2_X1, sFV_2_X2, sFV_X1X2, sFV_YX1, sFV_YX2);
  y += height;
  d_Gabor1 += height * d_Gabor1Pitch;
  component_velocity_two_frames(2, (const float2 *)d_Gabor1, x, y, op, n_valids,
                                sFV_2_X1, sFV_2_X2, sFV_X1X2, sFV_YX1, sFV_YX2);
  y += height;
  d_Gabor1 += height * d_Gabor1Pitch;
  component_velocity_two_frames(3, (const float2 *)d_Gabor1, x, y, op, n_valids,
                                sFV_2_X1, sFV_2_X2, sFV_X1X2, sFV_YX1, sFV_YX2);
  y += height;
  d_Gabor1 += height * d_Gabor1Pitch;
  component_velocity_two_frames(4, (const float2 *)d_Gabor1, x, y, op, n_valids,
                                sFV_2_X1, sFV_2_X2, sFV_X1X2, sFV_YX1, sFV_YX2);
  y += height;
  d_Gabor1 += height * d_Gabor1Pitch;
  component_velocity_two_frames(5, (const float2 *)d_Gabor1, x, y, op, n_valids,
                                sFV_2_X1, sFV_2_X2, sFV_X1X2, sFV_YX1, sFV_YX2);
  y += height;
  d_Gabor1 += height * d_Gabor1Pitch;
  component_velocity_two_frames(6, (const float2 *)d_Gabor1, x, y, op, n_valids,
                                sFV_2_X1, sFV_2_X2, sFV_X1X2, sFV_YX1, sFV_YX2);
  y += height;
  d_Gabor1 += height * d_Gabor1Pitch;
  component_velocity_two_frames(7, (const float2 *)d_Gabor1, x, y, op, n_valids,
                                sFV_2_X1, sFV_2_X2, sFV_X1X2, sFV_YX1, sFV_YX2);

  /*******************************/
  /* Intersection of Constraints */
  /*******************************/

  // Compute optic flow
  float invden = 1.0f;
  invden /= (sFV_2_X1 * sFV_2_X2 - sFV_X1X2 * sFV_X1X2) * TWO_PI * 0.25f;

  // Strictly speaking, the signs should be negated in the following (maybe the
  // filter orientations should be negative?)
  Ox = -(sFV_YX1 * sFV_2_X2 - sFV_YX2 * sFV_X1X2) * invden;
  Oy = -(sFV_YX2 * sFV_2_X1 - sFV_YX1 * sFV_X1X2) * invden;

  if (n_valids < 4) {
    Ox = unreliable;
    Oy = unreliable;
  }

  op.x += Ox; // add previous scale flow (doubled and interpolated)
  op.y += Oy; // add previous scale flow (doubled and interpolated)
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

// Core device optical flow computation (four orientations)

__device__ static void comp_optic_flow_two_frames_four_orientations(
    const char *d_Gabor1, int d_Gabor1Pitch, float2 &op, int x, int y,
    int height, float unreliable) {
  float Ox, Oy, sFV_YX1, sFV_YX2, sFV_2_X1, sFV_2_X2, sFV_X1X2;
  sFV_YX1 = 0.0f;
  sFV_YX2 = 0.0f;
  sFV_2_X1 = 0.0f;
  sFV_2_X2 = 0.0f;
  sFV_X1X2 = 0.0f;
  unsigned int n_valids = 0;

  /******************************/
  /* Component Velocity and MSE */
  /******************************/

  // unroll the loop

  d_Gabor1 += y * d_Gabor1Pitch + x * sizeof(float2);
  component_velocity_two_frames(0, (const float2 *)d_Gabor1, x, y, op, n_valids,
                                sFV_2_X1, sFV_2_X2, sFV_X1X2, sFV_YX1, sFV_YX2);
  y += height;
  d_Gabor1 += height * d_Gabor1Pitch;
  component_velocity_two_frames(2, (const float2 *)d_Gabor1, x, y, op, n_valids,
                                sFV_2_X1, sFV_2_X2, sFV_X1X2, sFV_YX1, sFV_YX2);
  y += height;
  d_Gabor1 += height * d_Gabor1Pitch;
  component_velocity_two_frames(4, (const float2 *)d_Gabor1, x, y, op, n_valids,
                                sFV_2_X1, sFV_2_X2, sFV_X1X2, sFV_YX1, sFV_YX2);
  y += height;
  d_Gabor1 += height * d_Gabor1Pitch;
  component_velocity_two_frames(6, (const float2 *)d_Gabor1, x, y, op, n_valids,
                                sFV_2_X1, sFV_2_X2, sFV_X1X2, sFV_YX1, sFV_YX2);

  /*******************************/
  /* Intersection of Constraints */
  /*******************************/

  // Compute optic flow
  float invden = 1.0f;
  invden /= (sFV_2_X1 * sFV_2_X2 - sFV_X1X2 * sFV_X1X2) * TWO_PI * 0.25f;

  // Strictly speaking, the signs should be negated in the following (maybe the
  // filter orientations should be negative?)
  Ox = -(sFV_YX1 * sFV_2_X2 - sFV_YX2 * sFV_X1X2) * invden;
  Oy = -(sFV_YX2 * sFV_2_X1 - sFV_YX1 * sFV_X1X2) * invden;

  if (n_valids < 4) {
    Ox = unreliable;
    Oy = unreliable;
  }

  op.x += Ox; // add previous scale flow (doubled and interpolated)
  op.y += Oy; // add previous scale flow (doubled and interpolated)
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

// Check if fetch location is within the image

__device__ static bool valid_fetch(float2 fetch, float width, float height) {

  return ((fetch.x >= 0.0f) & (fetch.x <= (width - 1.0f)) & (fetch.y >= 0.0f) &
          (fetch.y <= (height - 1.0f)));
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

// Two frame optical flow kernel

__global__ void optic_flow_two_frames_GPU(const char *d_Gabor1,
                                          int d_Gabor1Pitch,
                                          float2 *d_optic_flow, int width,
                                          int height, int d_optic_flowPitch,
                                          float unreliable, bool first_scale) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < width) & (y < height)) { // are we in the image?

    // fetch and transform previous scale optic flow (or init to zero at first
    // scale)
    float2 O = first_scale ? make_float2(0.0f, 0.0f)
                           : tex2D(d_o_prev_scale_texture, x * 0.5f + 0.5f,
                                   y * 0.5f + 0.5f);
    O.x *= 2.0f;
    O.y *= 2.0f;

    // check if flow warping remains inside the image

    // frame 2
    float2 fetch = make_float2((float)x + O.x, (float)y + O.y);
    if (valid_fetch(fetch, width, height)) {
      comp_optic_flow_two_frames(d_Gabor1, d_Gabor1Pitch, O, x, y, height,
                                 unreliable);
    } else {
      O.x += unreliable;
      O.y += unreliable;
    }

    // Save optic flow
    *((float2 *)((char *)d_optic_flow + y *d_optic_flowPitch) + x) = O;
  }
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

// Two frame optical flow kernel

__global__ void optic_flow_two_frames_four_orientations_GPU(
    const char *d_Gabor1, int d_Gabor1Pitch, float2 *d_optic_flow, int width,
    int height, int d_optic_flowPitch, float unreliable, bool first_scale) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < width) & (y < height)) { // are we in the image?

    // fetch and transform previous scale optic flow (or init to zero at first
    // scale)
    float2 O = first_scale ? make_float2(0.0f, 0.0f)
                           : tex2D(d_o_prev_scale_texture, x * 0.5f + 0.5f,
                                   y * 0.5f + 0.5f);
    O.x *= 2.0f;
    O.y *= 2.0f;

    // check if flow warping remains inside the image

    // frame 2
    float2 fetch = make_float2((float)x + O.x, (float)y + O.y);
    if (valid_fetch(fetch, width, height)) {
      comp_optic_flow_two_frames_four_orientations(d_Gabor1, d_Gabor1Pitch, O,
                                                   x, y, height, unreliable);
    } else {
      O.x += unreliable;
      O.y += unreliable;
    }

    // Save optic flow
    *((float2 *)((char *)d_optic_flow + y *d_optic_flowPitch) + x) = O;
  }
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

__global__ void flow_consist_GPU(float2 *d_flow1, float cons_thres, int width,
                                 int height, int pitch) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < width) & (y < height)) // are we in the image?
  {
    // Fetch frame 1 flow from global memory
    d_flow1 = ((float2 *)((char *)d_flow1 + y * pitch) + x);
    float2 flow1 = *d_flow1;

    // Fetch frame 2 flow (at pos + frame 1 flow) from texture
    float2 flow2 = tex2D(d_frame2_flow_texture, (float)x + flow1.x + 0.5f,
                         (float)y + flow1.y + 0.5f);

    // Check error

    float e_x = flow1.x + flow2.x;
    float e_y = flow1.y + flow2.y;

    float err = sqrtf(e_x * e_x + e_y * e_y);
    *d_flow1 = (err < cons_thres) ? flow1 : make_float2(nanf(""), nanf(""));
  }
}

////////////////////////
///  HOST FUNCTIONS  ///
////////////////////////

void compute_optical_flow_two_frames(
    std::vector<PitchFloat2Mem> &d_optic_flow_pyramid, char *d_TEMP,
    int d_TEMPPitch, const std::vector<PitchFloat2Mem> &gabPyr1_v,
    const std::vector<PitchFloat2Mem> &gabPyr2_v, int n_scales,
    bool median_filter, std::vector<int> &n_rows, std::vector<int> &n_cols,
    bool fourOrientations = false) {

  int N_ORIENTS = fourOrientations ? 4 : 8;

  // configuration
  dim3 dimBlock_flow(16, 8, 1);
  dim3 dimBlock_medfilt(16, 8, 1);

  // Setup textures
  cudaChannelFormatDesc channelFloat2 = cudaCreateChannelDesc<float2>();

  // Bind textures to lowest scale arrays
  const char *d_Gabor1 = (const char *)gabPyr1_v.at(0).ptr;
  int d_Gabor1Pitch = (int)gabPyr1_v.at(0).pitch;
  cudaBindTexture2D(0, &d_Gabor_texture2, gabPyr2_v.at(0).ptr, &channelFloat2,
                    n_cols[0], n_rows[0] * N_ORIENTS, gabPyr2_v.at(0).pitch);

  // compute first scale optical flow
  dim3 dimGrid_flow(iDivUp(n_cols[0], dimBlock_flow.x),
                    iDivUp(n_rows[0], dimBlock_flow.y), 1);

  if (median_filter) {
    if (fourOrientations)
      optic_flow_two_frames_four_orientations_GPU
              << <dimGrid_flow, dimBlock_flow>>>
          (d_Gabor1, d_Gabor1Pitch, (float2 *)d_TEMP, n_cols[0], n_rows[0],
           d_TEMPPitch, NAN_FLOAT, true);
    else
      optic_flow_two_frames_GPU << <dimGrid_flow, dimBlock_flow>>>
          (d_Gabor1, d_Gabor1Pitch, (float2 *)d_TEMP, n_cols[0], n_rows[0],
           d_TEMPPitch, NAN_FLOAT, true);
  } else {
    if (fourOrientations)
      optic_flow_two_frames_four_orientations_GPU
              << <dimGrid_flow, dimBlock_flow>>>
          (d_Gabor1, d_Gabor1Pitch, d_optic_flow_pyramid.at(0).ptr, n_cols[0],
           n_rows[0], (int)d_optic_flow_pyramid.at(0).pitch,
           (n_scales == 1) ? NAN_FLOAT : 0.0f, true);
    else
      optic_flow_two_frames_GPU << <dimGrid_flow, dimBlock_flow>>>
          (d_Gabor1, d_Gabor1Pitch, d_optic_flow_pyramid.at(0).ptr, n_cols[0],
           n_rows[0], (int)d_optic_flow_pyramid.at(0).pitch,
           (n_scales == 1) ? NAN_FLOAT : 0.0f, true);
  }

  // Coarse-to-fine processing
  for (int s = 1; s < n_scales; s++) {

    if (median_filter) {
      // 2D NaNMedianFilter the flow and turn NaNs into zeros
      cudaBindTexture2D(0, &d_o_prev_scale_texture, d_TEMP, &channelFloat2,
                        n_cols[s - 1], n_rows[s - 1], d_TEMPPitch);
      dim3 dimGrid_medfilt(iDivUp(n_cols[s - 1], dimBlock_medfilt.x),
                           iDivUp(n_rows[s - 1], dimBlock_medfilt.y), 1);
      nanmedfilt2_flow_GPU << <dimGrid_medfilt, dimBlock_medfilt>>>
          (d_optic_flow_pyramid.at(s - 1).ptr, n_rows[s - 1], n_cols[s - 1],
           (int)d_optic_flow_pyramid.at(s - 1).pitch, make_float2(0.0f, 0.0f));
    }
    cudaBindTexture2D(0, &d_o_prev_scale_texture,
                      d_optic_flow_pyramid.at(s - 1).ptr, &channelFloat2,
                      n_cols[s - 1], n_rows[s - 1],
                      d_optic_flow_pyramid.at(s - 1).pitch);

    // Bind textures to Gabor Pyramids
    d_Gabor1 = (const char *)gabPyr1_v.at(s).ptr;
    d_Gabor1Pitch = (int)gabPyr1_v.at(s).pitch;
    cudaBindTexture2D(0, &d_Gabor_texture2, gabPyr2_v.at(s).ptr, &channelFloat2,
                      n_cols[s], n_rows[s] * N_ORIENTS, gabPyr2_v.at(s).pitch);

    ///////////////////////
    // Update optic flow //
    ///////////////////////

    dim3 dimGrid_flow(iDivUp(n_cols[s], dimBlock_flow.x),
                      iDivUp(n_rows[s], dimBlock_flow.y), 1);

    if (median_filter) {
      if (fourOrientations)
        optic_flow_two_frames_four_orientations_GPU
                << <dimGrid_flow, dimBlock_flow>>>
            (d_Gabor1, d_Gabor1Pitch, (float2 *)d_TEMP, n_cols[s], n_rows[s],
             d_TEMPPitch, NAN_FLOAT, false);
      else
        optic_flow_two_frames_GPU << <dimGrid_flow, dimBlock_flow>>>
            (d_Gabor1, d_Gabor1Pitch, (float2 *)d_TEMP, n_cols[s], n_rows[s],
             d_TEMPPitch, NAN_FLOAT, false);
    } else {
      if (fourOrientations)
        optic_flow_two_frames_four_orientations_GPU
                << <dimGrid_flow, dimBlock_flow>>>
            (d_Gabor1, d_Gabor1Pitch, d_optic_flow_pyramid.at(s).ptr, n_cols[s],
             n_rows[s], (int)d_optic_flow_pyramid.at(s).pitch,
             (s == (n_scales - 1)) ? NAN_FLOAT : 0.0f, false);
      else
        optic_flow_two_frames_GPU << <dimGrid_flow, dimBlock_flow>>>
            (d_Gabor1, d_Gabor1Pitch, d_optic_flow_pyramid.at(s).ptr, n_cols[s],
             n_rows[s], (int)d_optic_flow_pyramid.at(s).pitch,
             (s == (n_scales - 1)) ? NAN_FLOAT : 0.0f, false);
    }

  } //     for(int s=1;s<n_scales;s++)

  if (median_filter) {
    // Median filter final flow (now keeping NaNs)
    cudaBindTexture2D(0, &d_o_prev_scale_texture, d_TEMP, &channelFloat2,
                      n_cols[n_scales - 1], n_rows[n_scales - 1], d_TEMPPitch);
    dim3 dimGrid_medfilt(iDivUp(n_cols[n_scales - 1], dimBlock_medfilt.x),
                         iDivUp(n_rows[n_scales - 1], dimBlock_medfilt.y), 1);
    nanmedfilt2_flow_GPU << <dimGrid_medfilt, dimBlock_medfilt>>>
        (d_optic_flow_pyramid.at(n_scales - 1).ptr, n_rows[n_scales - 1],
         n_cols[n_scales - 1], (int)d_optic_flow_pyramid.at(n_scales - 1).pitch,
         make_float2(NAN_FLOAT, NAN_FLOAT));
  }

  //  printf("%s\n",cudaGetErrorString(cudaGetLastError()));
}

void compute_consistent_optical_flow_two_frames(
    std::vector<PitchFloat2Mem> &d_optic_flow_pyramid, char *d_TEMP,
    int d_TEMPPitch, const std::vector<PitchFloat2Mem> &gabPyr1_v,
    const std::vector<PitchFloat2Mem> &gabPyr2_v,
    cudaArray *d_frame2_flow_array, int n_scales, bool median_filter,
    std::vector<int> &n_rows, std::vector<int> &n_cols, float cons_thres,
    bool fourOrientations = false) {

  // Compute flow 2 -> 1, situated in 2
  compute_optical_flow_two_frames(d_optic_flow_pyramid, d_TEMP, d_TEMPPitch,
                                  gabPyr2_v, gabPyr1_v, n_scales, median_filter,
                                  n_rows, n_cols, fourOrientations);

  // Copy to array for consistency check
  cudaMemcpy2DToArray(d_frame2_flow_array, 0, 0,
                      d_optic_flow_pyramid.at(n_scales - 1).ptr,
                      d_optic_flow_pyramid.at(n_scales - 1).pitch,
                      n_cols[n_scales - 1] * sizeof(float2),
                      n_rows[n_scales - 1], cudaMemcpyDeviceToDevice);

  // Compute flow 1 -> 2, situated in 1
  compute_optical_flow_two_frames(d_optic_flow_pyramid, d_TEMP, d_TEMPPitch,
                                  gabPyr1_v, gabPyr2_v, n_scales, median_filter,
                                  n_rows, n_cols, fourOrientations);

  // Consistency check
  cudaChannelFormatDesc channelFloat2 = cudaCreateChannelDesc<float2>();
  cudaBindTextureToArray(d_frame2_flow_texture, d_frame2_flow_array,
                         channelFloat2);

  dim3 dimBlock_flow(16, 16, 1);
  dim3 dimGrid_flow(iDivUp(n_cols[n_scales - 1], dimBlock_flow.x),
                    iDivUp(n_rows[n_scales - 1], dimBlock_flow.y), 1);
  flow_consist_GPU << <dimGrid_flow, dimBlock_flow>>>
      (d_optic_flow_pyramid.at(n_scales - 1).ptr, cons_thres,
       n_cols[n_scales - 1], n_rows[n_scales - 1],
       (int)d_optic_flow_pyramid.at(n_scales - 1).pitch);

  //  flow_consist_GPU<<<dimGrid_flow,dimBlock_flow>>>(d_optic_flow_pyramid[n_scales-1],
  // cons_thres, n_cols[n_scales-1], n_rows[n_scales-1],
  // (int)d_optic_flow_pyramidPitch[n_scales-1]);
}

///////////////////////////////////////////////////////////////
// Calling function
///////////////////////////////////////////////////////////////

void computeOpticalFlowTwoFrames(
    std::vector<PitchFloat2Mem> &d_optic_flow_pyramid, char *d_TEMP,
    int d_TEMPPitch, const std::vector<PitchFloat2Mem> &gabPyr1_v,
    const std::vector<PitchFloat2Mem> &gabPyr2_v,
    cudaArray *d_frame2_flow_array, int n_scales, bool median_filter,
    bool consistent, float cons_thres, std::vector<int> &n_rows,
    std::vector<int> &n_cols, bool fourOrientations) {

  d_Gabor_texture2.addressMode[0] = cudaAddressModeClamp;
  d_Gabor_texture2.addressMode[1] = cudaAddressModeClamp;
  d_Gabor_texture2.filterMode = cudaFilterModeLinear;
  d_Gabor_texture2.normalized = false;

  d_o_prev_scale_texture.addressMode[0] = cudaAddressModeClamp;
  d_o_prev_scale_texture.addressMode[1] = cudaAddressModeClamp;
  d_o_prev_scale_texture.filterMode = cudaFilterModeLinear;
  d_o_prev_scale_texture.normalized = false;

  d_frame2_flow_texture.addressMode[0] = cudaAddressModeClamp;
  d_frame2_flow_texture.addressMode[1] = cudaAddressModeClamp;
  d_frame2_flow_texture.filterMode = cudaFilterModeLinear;
  d_frame2_flow_texture.normalized = false;

  if (consistent)
    compute_consistent_optical_flow_two_frames(
        d_optic_flow_pyramid, d_TEMP, d_TEMPPitch, gabPyr1_v, gabPyr2_v,
        d_frame2_flow_array, n_scales, median_filter, n_rows, n_cols,
        cons_thres, fourOrientations);
  else
    compute_optical_flow_two_frames(
        d_optic_flow_pyramid, d_TEMP, d_TEMPPitch, gabPyr1_v, gabPyr2_v,
        n_scales, median_filter, n_rows, n_cols, fourOrientations);
}

} // end namespace vision
