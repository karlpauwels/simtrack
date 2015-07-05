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
#include <math.h>
#include <cuda_runtime.h>

namespace vision {

const float NAN_FLOAT = nanf("");

// Round a / b to nearest higher integer value
int iDivUp(int a, int b);

// convert float into jet colormap image

void convertFloatToRGBA(uchar4 *d_out_image, const float *d_in_image, int width,
                        int height, float lowerLim, float upperLim);
void convertPitchedFloatToRGBA(uchar4 *d_out_image, const float *d_in_image,
                               int width, int height, int pitch, float lowerLim,
                               float upperLim);
void convertFloatToRGBA(uchar4 *d_out_image, const float *d_in_image, int width,
                        int height);

// transforms pitched float image into RGBA grayscale
void convertPitchedFloatToGrayRGBA(uchar4 *d_out_image, const float *d_in_image,
                                   int width, int height, int pitch,
                                   float lowerLim, float upperLim);

// same but with array
void convertFloatArrayToGrayRGBA(uchar4 *d_out_image, cudaArray *in_array,
                                 int width, int height, float lower_lim = 0.0f,
                                 float upper_lim = 1.0f);

void convertKinectFloatToRGBA(uchar4 *d_out_image, const float *d_in_image,
                              int width, int height, int pitch, float lowerLim,
                              float upperLim);

// converts flow to rgba, discarding below minMag magnitude

void convertFlowToRGBA(uchar4 *d_flowx_out, uchar4 *d_flowy_out,
                       const float *d_flowx_in, const float *d_flowy_in,
                       int width, int height, float lowerLim, float upperLim,
                       float minMag);

// creates an RGBA anaglyph applying pre_shift to the right image

void createAnaglyph(uchar4 *d_out_image, const float *d_left_image,
                    const float *d_right_image, int width, int height,
                    int pre_shift);

void createAnaglyph(uchar4 *d_out_image, const uchar4 *d_left_image,
                    const uchar4 *d_right_image, int width, int height,
                    int pre_shift);

// convert 2D vectors to an RGBA angle image and an RGBA magnitude image
// lower_ang and upper_ang are in radians

void convert2DVectorToAngleMagnitude(uchar4 *d_angle_image,
                                     uchar4 *d_magnitude_image,
                                     float *d_vector_X, float *d_vector_Y,
                                     int width, int height, float lower_ang,
                                     float upper_ang, float lower_mag,
                                     float upper_mag);

// threshold floats using lowerLim and upperLim into RGBA black/white image

void convertFloatToRGBAbinary(uchar4 *out_image, const float *in_image,
                              int width, int height, float lowerLim,
                              float upperLim);

// blend float image with float label (membership specified by lowerLim and
// upperLim)

void blendFloatImageFloatLabelToRGBA(uchar4 *out_image, const float *in_image,
                                     const float *label, int width, int height,
                                     float lowerLim, float upperLim);

void blendFloatImageRGBAArrayToRGBA(uchar4 *out_image, const float *in_image,
                                    cudaArray *in_array, int width, int height,
                                    float w_r, float w_g, float w_b);

void blendFloatImageFloatArrayToRGBA(uchar4 *out_image, const float *in_image,
                                     cudaArray *in_array, int pitch_out,
                                     int pitch_in, int width, int height);

void blendMultiColor(uchar4 *out_image, const float *in_image,
                     cudaArray *in_texture, cudaArray *in_segment_index,
                     int width, int height);

// void augmentedReality(uchar4 *out_image, const float *in_image, cudaArray
// *in_array, int width, int height);

void augmentedReality(float *out_image, const float *in_image,
                      cudaArray *in_array, int width, int height, int pitch);

void augmentedRealityFloatArray(float *out_image, const float *in_image,
                                cudaArray *in_array, int width, int height,
                                int pitch);

void augmentedRealityFloatArraySelectiveBlend(float *out_image,
                                              const float *in_image,
                                              const cudaArray *texture,
                                              const cudaArray *segment_index,
                                              int width, int height, int pitch,
                                              int max_segment_ind);

// remove all flow and disparity that is not mutually valid

void mutuallyValidateFlowStereo(float *d_flow, float *d_disparity, int width,
                                int height);

// de-interleave

void deInterleave(float *d_X_out, float *d_Y_out, float2 *d_XY_in,
                  int pitch_out, int pitch_in, int width, int height);

void applyIMOMask(float *d_IMOMask, float *d_IMO, const float *d_disparity,
                  float offset, int width, int height);

void invalidateFlow(float *modFlowX, float *modFlowY, const float *constFlowX,
                    const float *constFlowY, int width, int height,
                    float cons_thres);

void colorInvalids(uchar4 *out_image, const float *in_image, int width,
                   int height);

void colorBlend(uchar4 *out_image, const uchar4 *in_image, cudaArray *in_array,
                int width, int height, float alpha_scale);

void convertKinectDisparityToRegularDisparity(float *d_regularDisparity,
                                              int d_regularDisparityPitch,
                                              const float *d_KinectDisparity,
                                              int d_KinectDisparityPitch,
                                              int width, int height);

void convertKinectDisparityInPlace(float *d_disparity, int pitch, int width,
                                   int height, float depth_scale);

void colorDistDiff(uchar4 *out_image, const float *disparity,
                   int disparity_pitch, const float *disparity_prior, int width,
                   int height, float focal_length, float baseline,
                   float nodal_point_x, float nodal_point_y, float dist_thres);

} // end namespace vision
