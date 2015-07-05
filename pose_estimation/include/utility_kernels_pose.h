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

#include <cuda_runtime.h>

namespace pose {

int divUp(int a, int b);

void get_GL_conv_constants(float &Z_conv1, float &Z_conv2, float far_plane,
                           float near_plane);

void convertZbufferToZ(float *d_Z, cudaArray *d_ZbufferArray, int n_cols,
                       int n_rows, float nodal_point_x, float nodal_point_y,
                       float near_plane, float far_plane);

void convertZbufferToDisparity(float *d_Disparity, cudaArray *d_ZbufferArray,
                               int n_cols, int n_rows, int pitch,
                               float nodal_point_x, float nodal_point_y,
                               float near_plane, float far_plane,
                               float focal_length, float baseline);

void save_device_var_to_file(const char *file_name, const void *device_var,
                             int elem_size, int n_elements);

float approx_median_shuffle_cuda(float *d_data, float *d_tmp,
                                 float *d_random_numbers, int pp);

void approx_multiple_medians_shuffle_cuda(float *medians, float *d_data,
                                          float *d_tmp,
                                          const float *d_random_numbers,
                                          const int *pp, int n_segments,
                                          int *d_n_in, int *d_start_inds);

void convertPointCloudToDepthImage(unsigned int *d_depth_image,
                                   const float4 *d_point_cloud, int n_cols,
                                   int n_rows, int n_points,
                                   float nodal_point_x, float nodal_point_y,
                                   float focal_length_x, float focal_length_y,
                                   const float *d_translation_vector,
                                   const float *d_rotation_matrix);

void convertDepthImageToMeter(float *d_depth_image_meter,
                              const unsigned int *d_depth_image_millimeter,
                              int n_cols, int n_rows);

void colorValidationDepthImageMatches(uchar4 *out_image,
                                      const float *d_depth_image,
                                      cudaArray *d_z_buffer_array, int width,
                                      int height, float near_plane,
                                      float far_plane, float max_error,
                                      float llim_depth, float ulim_depth);

/*! \brief Engine constructor
    \param starting_indices  output array: starting indices for each label (note that
   0 is also considered a label)
                        assumed pre-allocated and of length (max_label+1)
                        will be set to -1 in absence of label
    \param labels       sorted array of device labels
    \param n_labels     length of labels array
    \param max_label    largest label
 */
void extractLabelStartingIndices(int *starting_indices, unsigned int *labels,
                                 int n_labels, int max_label);
}
