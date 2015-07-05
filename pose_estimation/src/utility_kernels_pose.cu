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
#include <utility_kernels_pose.h>

namespace pose {

// number of median reductions performed within each block
// implies that we have 3^MED_BLOCK_LEVELS threads
const int MED_BLOCK_LEVELS = 5;
const int MED_BLOCK_SIZE = 243;

texture<float, cudaTextureType2D, cudaReadModeElementType> d_Zbuffer_texture;
texture<unsigned int, 1, cudaReadModeElementType> labelsTexture;

// Convert Zbuffer for initial Z model construction
static __global__ void convert_Zbuffer_to_Z_GPU(float *d_Z, int n_cols,
                                                int n_rows, float Z_conv1,
                                                float Z_conv2, float floatnan) {

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < n_cols) & (y < n_rows)) // are we in the image?
  {
    // determine output linear index
    unsigned int ind = x + y * n_cols;

    // determine gl coord
    float xt = (float)x + 0.5f;
    float yt = (float)y + 0.5f;

    float Zbuffer = tex2D(d_Zbuffer_texture, xt, yt);

    d_Z[ind] =
        (Zbuffer > 0.0f) ? (__fdividef(Z_conv1, Zbuffer + Z_conv2)) : floatnan;
  }
}

// Convert Zbuffer to disparity
static __global__ void
convert_Zbuffer_to_Disparity_GPU(float *d_Disparity, int n_cols, int n_rows,
                                 int pitch, float D_conv1, float D_conv2,
                                 float floatnan) {

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < n_cols) & (y < n_rows)) // are we in the image?
  {
    // determine output linear index
    //    unsigned int ind = x + y*n_cols;

    // determine gl coord
    float xt = (float)x + 0.5f;
    float yt = (float)y + 0.5f;

    float Zbuffer = tex2D(d_Zbuffer_texture, xt, yt);

    *((float *)((char *)d_Disparity + y *pitch) + x) =
        (Zbuffer > 0.0f) ? (D_conv1 * Zbuffer + D_conv2) : floatnan;
  }
}

__global__ void convertPointCloudToDepthImage_kernel(
    unsigned int *depth_image, const float4 *point_cloud, int n_cols,
    int n_rows, int n_points, float nodal_point_x, float nodal_point_y,
    float focal_length_x, float focal_length_y, const float *T,
    const float *R) {

  const int ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n_points) {

    // fetch point
    float4 point = point_cloud[ind];

    // transform to camera frame
    float x = R[0] * point.x + R[1] * point.y + R[2] * point.z + T[0];
    float y = R[3] * point.x + R[4] * point.y + R[5] * point.z + T[1];
    float z = R[6] * point.x + R[7] * point.y + R[8] * point.z + T[2];

    float inv_z = 1.0f / z;

    // project in image
    int x_pix = __float2int_rn(focal_length_x * x * inv_z + nodal_point_x);
    int y_pix = __float2int_rn(focal_length_y * y * inv_z + nodal_point_y);

    // check if inside image
    bool valid =
        ((x_pix >= 0) && (x_pix < n_cols) && (y_pix >= 0) && (y_pix < n_rows));

    if (valid) {
      int ind_out = y_pix * n_cols + x_pix;
      //      depth_image[ind_out] = (unsigned int)(point.z * 1000.0f);
      atomicMin(depth_image + ind_out, (unsigned int)(point.z * 1000.0f));
    }
  }
}

__global__ void initializeToValue_kernel(unsigned int *data, unsigned int value,
                                         int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    data[y * width + x] = value;
  }
}

__global__ void
convertDepthImageToMeter_kernel(float *d_depth_image_meter,
                                const unsigned int *d_depth_image_millimeter,
                                int n_rows, int n_cols) {

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < n_cols && y < n_rows) {
    int ind = y * n_cols + x;
    unsigned int depth = d_depth_image_millimeter[ind];
    d_depth_image_meter[ind] =
        (depth == 4294967295) ? nanf("") : (float)depth / 1000.0f;
  }
}

__global__ void colorValidationDepthImageMatches_kernel(
    uchar4 *out_image, const float *depth_image, int width, int height,
    float Z_conv1, float Z_conv2, float max_error, float llim_depth,
    float ulim_depth) {

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < width) & (y < height)) // are we in the image?
  {
    // determine i/o linear index
    unsigned int ind = x + y * width;

    // determine gl coord
    float xt = (float)x + 0.5f;
    float yt = (float)y + 0.5f;

    float Zbuffer = tex2D(d_Zbuffer_texture, xt, yt);

    bool model_present = Zbuffer > 0.0f;

    float Z_measured = depth_image[ind];

    float Z_estimated = __fdividef(Z_conv1, Zbuffer + Z_conv2);

    bool validated = fabsf(Z_estimated - Z_measured) <= max_error;

    unsigned char output_intensity =
        255 * (Z_measured - llim_depth) / (ulim_depth - llim_depth);

    uchar4 outpixel;

    outpixel.x = output_intensity;
    outpixel.y = output_intensity;
    outpixel.z = output_intensity;
    outpixel.w = 255;

    if (model_present) {
      outpixel.x = validated ? 0 : output_intensity;
      outpixel.y = validated ? output_intensity : 0;
      outpixel.z = 0;
    }

    out_image[ind] = outpixel;
  }
}

// Approximate median with data shuffling
// The input data is shuffled (with replacement) according to the random
// numbers, n_in is the size of the input data (no limit)
__global__ void median_reduce_shuffle_gpu(const float *d_in, float *d_out,
                                          float *d_random_numbers, int n_in) {

  /**************/
  /* initialize */
  /**************/

  // compute indices

  int t_ind = threadIdx.x;
  int g_ind = blockIdx.x * MED_BLOCK_SIZE + t_ind;

  // allocate shared memory

  __shared__ float DATA[MED_BLOCK_SIZE];

  /**************/
  /* load stage */
  /**************/

  int sample_ind = floorf(d_random_numbers[g_ind] * (float)n_in);
  DATA[t_ind] = d_in[sample_ind];

  __syncthreads();

  /*******************/
  /* reduction stage */
  /*******************/

  for (int s = 1; s < MED_BLOCK_SIZE; s *= 3) {

    int index = 3 * s * t_ind;

    if (index < MED_BLOCK_SIZE) {

      // fetch three values
      float value1 = DATA[index];
      float value2 = DATA[index + s];
      float value3 = DATA[index + 2 * s];

      // extract the middle value (median)
      float smallest = fminf(value1, value2);
      value2 = fmaxf(value1, value2);
      value1 = smallest;

      value3 = fmaxf(value1, value3);
      value2 = fminf(value2, value3);

      DATA[index] = value2;
    }

    __syncthreads();
  }

  /***************/
  /* write stage */
  /***************/

  // write this block's approx median (first element)

  if (t_ind == 0) {
    d_out[blockIdx.x] = DATA[0];
  }
}

// Approximate median with data shuffling
// The input data is shuffled (with replacement) according to the random
// numbers, n_in is the size of the input data (no limit)
__global__ void
multiple_median_reduce_shuffle_gpu(const float *d_in, float *d_out,
                                   const float *d_random_numbers,
                                   const int *d_start_inds, const int *d_n_in) {

  /**************/
  /* initialize */
  /**************/

  int segment = blockIdx.y;

  // compute indices

  int t_ind = threadIdx.x;
  int g_ind =
      blockIdx.x * MED_BLOCK_SIZE +
      t_ind; // means that every row of blocks uses the same random numbers

  // allocate shared memory

  //  __shared__ float DATA[MED_BLOCK_SIZE];
  __shared__ float DATA[256];

  /**************/
  /* load stage */
  /**************/

  if (t_ind < MED_BLOCK_SIZE) {
    int sample_ind = d_start_inds[segment] +
                     floorf(d_random_numbers[g_ind] * (float)d_n_in[segment]);
    DATA[t_ind] = d_in[sample_ind];
  }

  __syncthreads();

  /*******************/
  /* reduction stage */
  /*******************/

  for (int s = 1; s < MED_BLOCK_SIZE; s *= 3) {

    int index = 3 * s * t_ind;

    if (index < MED_BLOCK_SIZE) {

      // fetch three values
      float value1 = DATA[index];
      float value2 = DATA[index + s];
      float value3 = DATA[index + 2 * s];

      // extract the middle value (median)
      float smallest = fminf(value1, value2);
      value2 = fmaxf(value1, value2);
      value1 = smallest;

      value3 = fmaxf(value1, value3);
      value2 = fminf(value2, value3);

      DATA[index] = value2;
    }

    __syncthreads();
  }

  /***************/
  /* write stage */
  /***************/

  // write this block's approx median (first element)

  if (t_ind == 0) {
    d_out[gridDim.x * blockIdx.y + blockIdx.x] = DATA[0];
  }
}

// writes the index of the first occurence of label l in labels into
// starting_indices

__global__ void extractLabelStartingIndicesGPU(int *starting_indices,
                                               unsigned int *labels,
                                               int n_labels) {

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < n_labels) {

    unsigned int curr_label = tex1Dfetch(labelsTexture, x);

    if ((x == 0) || (curr_label != tex1Dfetch(labelsTexture, x - 1)))
      starting_indices[curr_label] = x;
  }
}

///////////////////////
//                   //
// Calling functions //
//                   //
///////////////////////

int divUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// void get_GL_conv_constants(float &Z_conv1, float &Z_conv2, int n_cols, int
// n_rows, float nodal_point_x, float nodal_point_y, float far_plane, float
// near_plane) {
void get_GL_conv_constants(float &Z_conv1, float &Z_conv2, float far_plane,
                           float near_plane) {

  double f = (double)(far_plane);
  double n = (double)(near_plane);

  Z_conv1 = (float)((-f * n) / (f - n));
  Z_conv2 = (float)(-(f + n) / (2 * (f - n)) - 0.5);
}

void convertZbufferToZ(float *d_Z, cudaArray *d_ZbufferArray, int n_cols,
                       int n_rows, float nodal_point_x, float nodal_point_y,
                       float near_plane, float far_plane) {

  // Determine Zbuffer conversion constants
  // depth = Z_conv1/(Zbuffer+Z_conv2)
  float Z_conv1, Z_conv2;
  get_GL_conv_constants(Z_conv1, Z_conv2, far_plane, near_plane);

  // Bind textures to arrays
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();
  cudaBindTextureToArray(d_Zbuffer_texture, d_ZbufferArray, channelFloat);

  // Generate nan;
  //  unsigned long raw = 0x7FFFFFFF;
  //  float floatnan = *(float *)&raw;

  float floatnan = nanf("");

  // Convert Zbuffer
  dim3 TB(16, 16, 1);
  dim3 BG(divUp(n_cols, TB.x), divUp(n_rows, TB.y));
  convert_Zbuffer_to_Z_GPU << <BG, TB>>>
      (d_Z, n_cols, n_rows, Z_conv1, Z_conv2, floatnan);
}

void convertZbufferToDisparity(float *d_Disparity, cudaArray *d_ZbufferArray,
                               int n_cols, int n_rows, int pitch,
                               float nodal_point_x, float nodal_point_y,
                               float near_plane, float far_plane,
                               float focal_length, float baseline) {

  // Determine Zbuffer conversion constants
  // depth = Z_conv1/(Zbuffer+Z_conv2)
  float Z_conv1, Z_conv2;
  get_GL_conv_constants(Z_conv1, Z_conv2, far_plane, near_plane);

  // modify conversion constants
  float D_conv1 = (float)((double)(-focal_length * baseline) / (double)Z_conv1);
  float D_conv2 = D_conv1 * Z_conv2;

  // Bind textures to arrays
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();
  cudaBindTextureToArray(d_Zbuffer_texture, d_ZbufferArray, channelFloat);

  // Generate nan;
  unsigned long raw = 0x7FFFFFFF;
  float floatnan = *(float *)&raw;

  // Convert Zbuffer
  dim3 TB(16, 16, 1);
  dim3 BG(divUp(n_cols, TB.x), divUp(n_rows, TB.y));
  convert_Zbuffer_to_Disparity_GPU << <BG, TB>>>
      (d_Disparity, n_cols, n_rows, pitch, D_conv1, D_conv2, floatnan);
}

void save_device_var_to_file(const char *file_name, const void *device_var,
                             int elem_size, int n_elements) {

  void *host_var = malloc(n_elements * elem_size);
  FILE *fout = fopen(file_name, "wb");
  cudaMemcpy(host_var, device_var, n_elements * elem_size,
             cudaMemcpyDeviceToHost);
  fwrite(host_var, elem_size, n_elements, fout);
  fclose(fout);
  free(host_var);
}

// Compute the 1D approximate median using the first power of 3 elements
// d_data gets overwritten!
// the data gets shuffled (with replacement) at each stage
float approx_median_shuffle_cuda(float *d_data, float *d_tmp,
                                 float *d_random_numbers, int pp) {

  // Make sure we're working with a power of 3 number of elements
  // (select the first power of 3 elements)
  int n_levels = (int)floor(log(double(pp)) / log(3.0));

  // Define block size (grid size defined later)
  dim3 threadBlock(MED_BLOCK_SIZE, 1);

  int n_blocks;
  float *d_out;
  float *d_in;

  // d_data and d_tmp are constantly swapped -> d_data gets overwritten!
  d_in = d_data; // initial input is the data matrix
  d_out = d_tmp; // initial output is d_tmp

  int n_in = pp; // initially all data can be used for sampling

  while (n_levels > 0) {

    // Number of blocks and elements in block (only smaller for last block)
    n_blocks = 1;
    if (n_levels >= MED_BLOCK_LEVELS)
      n_blocks = (int)pow(3.0, n_levels - MED_BLOCK_LEVELS);

    dim3 blockGrid(n_blocks, 1);
    median_reduce_shuffle_gpu << <blockGrid, threadBlock>>>
        (d_in, d_out, d_random_numbers, n_in);

    // Update variables for next iteration
    n_levels -= MED_BLOCK_LEVELS;
    n_in = (int)pow(3.0, n_levels);

    // Swap input and output pointers
    float *tmp = d_in;
    d_in = d_out;
    d_out = tmp;
  }

  // Copy result back to host
  // (the last reduction output was swapped into d_in)

  float median;
  cudaMemcpy(&median, d_in, sizeof(float), cudaMemcpyDeviceToHost);

  return (median);
}

// Compute the 1D approximate medians using the first power of 3 elements
// d_data gets overwritten!
// the data gets shuffled (with replacement) at each stage
// this version handles multiple datastreams packed in d_data
// pp contains the number of values for each segment
void approx_multiple_medians_shuffle_cuda(float *medians, float *d_data,
                                          float *d_tmp,
                                          const float *d_random_numbers,
                                          const int *pp, int n_segments,
                                          int *d_n_in, int *d_start_inds) {

  // Make sure we're working with a power of 3 number of elements
  // Use n_levels determined by largest stream
  int pp_max = pp[0];
  for (int i = 1; i < n_segments; i++)
    if (pp[i] > pp_max)
      pp_max = pp[i];

  int n_levels = (int)floor(log(double(pp_max)) / log(3.0));

  // never process more than 3^9 elements per segment = 19683

  n_levels = (n_levels > 9) ? 9 : n_levels;

  int start_inds[n_segments];
  start_inds[0] = 0;
  for (int i = 1; i < n_segments; i++)
    start_inds[i] = start_inds[i - 1] + pp[i - 1];

  // Define block size (grid size defined later)
  //  dim3 threadBlock(MED_BLOCK_SIZE,1);
  dim3 threadBlock(256, 1);

  int n_blocks;
  float *d_out;
  float *d_in;

  // d_data and d_tmp are constantly swapped -> d_data gets overwritten!
  d_in = d_data; // initial input is the data matrix
  d_out = d_tmp; // initial output is d_tmp

  // initially all data can be used for sampling
  int n_in[n_segments];
  memcpy(n_in, pp, n_segments * sizeof(int));
  //  int *d_n_in, *d_start_inds;
  //  cudaMalloc((void**)&d_n_in,n_segments*sizeof(int));
  //  cudaMalloc((void**)&d_start_inds,n_segments*sizeof(int));
  cudaMemcpy(d_n_in, n_in, n_segments * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_start_inds, start_inds, n_segments * sizeof(int),
             cudaMemcpyHostToDevice);

  //  printf("starting med it\n");

  while (n_levels > 0) {

    // Number of blocks and elements in block (only smaller for last block)
    n_blocks = 1;
    if (n_levels > MED_BLOCK_LEVELS)
      n_blocks = (int)pow(3.0, n_levels - MED_BLOCK_LEVELS);

    dim3 blockGrid(n_blocks, n_segments);

    //    printf("blockgrid: %d %d\n",blockGrid.x,blockGrid.y);

    multiple_median_reduce_shuffle_gpu << <blockGrid, threadBlock>>>
        (d_in, d_out, d_random_numbers, d_start_inds, d_n_in);

    // Update variables for next iteration
    n_levels -= MED_BLOCK_LEVELS;

    int n = (int)pow(3.0, n_levels);
    for (int i = 0; i < n_segments; i++) {
      n_in[i] = n;
      start_inds[i] = i * n;
    }
    cudaMemcpy(d_n_in, n_in, n_segments * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_inds, start_inds, n_segments * sizeof(int),
               cudaMemcpyHostToDevice);

    // Swap input and output pointers
    float *tmp = d_in;
    d_in = d_out;
    d_out = tmp;
  }

  // Copy result back to host
  // (the last reduction output was swapped into d_in)

  cudaMemcpy(medians, d_in, n_segments * sizeof(float), cudaMemcpyDeviceToHost);

  //  cudaFree(d_start_inds);
  //  cudaFree(d_n_in);
}

void convertPointCloudToDepthImage(unsigned int *d_depth_image,
                                   const float4 *d_point_cloud, int n_cols,
                                   int n_rows, int n_points,
                                   float nodal_point_x, float nodal_point_y,
                                   float focal_length_x, float focal_length_y,
                                   const float *d_translation_vector,
                                   const float *d_rotation_matrix) {

  // initialize depth_image to max value
  dim3 dim_block_init(16, 8);
  dim3 dim_grid_init(divUp(n_cols, dim_block_init.x),
                     divUp(n_rows, dim_block_init.y));
  initializeToValue_kernel << <dim_grid_init, dim_block_init>>>
      (d_depth_image, 4294967295, n_cols, n_rows);

  dim3 dim_block(128);
  dim3 dim_grid(divUp(n_points, dim_block.x));

  convertPointCloudToDepthImage_kernel << <dim_grid, dim_block>>>
      (d_depth_image, d_point_cloud, n_cols, n_rows, n_points, nodal_point_x,
       nodal_point_y, focal_length_x, focal_length_y, d_translation_vector,
       d_rotation_matrix);
}

void convertDepthImageToMeter(float *d_depth_image_meter,
                              const unsigned int *d_depth_image_millimeter,
                              int n_cols, int n_rows) {

  dim3 dim_block(16, 8);
  dim3 dim_grid(divUp(n_cols, dim_block.x), divUp(n_rows, dim_block.y));
  convertDepthImageToMeter_kernel << <dim_grid, dim_block>>>
      (d_depth_image_meter, d_depth_image_millimeter, n_rows, n_cols);
}

void colorValidationDepthImageMatches(uchar4 *out_image,
                                      const float *d_depth_image,
                                      cudaArray *d_z_buffer_array, int width,
                                      int height, float near_plane,
                                      float far_plane, float max_error,
                                      float llim_depth, float ulim_depth) {

  dim3 dimBlock(16, 8);
  dim3 dimGrid(divUp(width, dimBlock.x), divUp(height, dimBlock.y));

  // Determine Zbuffer conversion constants
  // depth = Z_conv1/(Zbuffer+Z_conv2)
  float Z_conv1, Z_conv2;
  get_GL_conv_constants(Z_conv1, Z_conv2, far_plane, near_plane);

  // Bind textures to arrays
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();
  cudaBindTextureToArray(d_Zbuffer_texture, d_z_buffer_array, channelFloat);

  colorValidationDepthImageMatches_kernel << <dimGrid, dimBlock>>>
      (out_image, d_depth_image, width, height, Z_conv1, Z_conv2, max_error,
       llim_depth, ulim_depth);
}

void extractLabelStartingIndices(int *starting_indices, unsigned int *labels,
                                 int n_labels, int max_label) {

  cudaMemset(starting_indices, -1, (max_label + 1) * sizeof(int));

  labelsTexture.normalized = 0;
  labelsTexture.filterMode = cudaFilterModePoint;
  labelsTexture.addressMode[0] = cudaAddressModeClamp;

  cudaChannelFormatDesc channelUINT = cudaCreateChannelDesc<unsigned int>();

  cudaBindTexture(0, &labelsTexture, labels, &channelUINT);

  dim3 threads(256);
  dim3 blocks(divUp(n_labels, threads.x));

  extractLabelStartingIndicesGPU << <blocks, threads>>>
      (starting_indices, labels, n_labels);

  cudaUnbindTexture(labelsTexture);
}
}
