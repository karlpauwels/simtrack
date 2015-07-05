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
#include <convolution_kernels.h>

namespace vision {

texture<float, 2, cudaReadModeElementType> imageTexture;
texture<float, 2, cudaReadModeElementType> floatTexture;
texture<float2, 2, cudaReadModeElementType> float2Texture;

// 24-bit multiplication is faster on G80,
// but we must be sure to multiply integers
// only within [-8M, 8M - 1] range
#define IMUL(a, b) __mul24(a, b)
// Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) (__mul24((a), (b)) + (c))

// image resize kernel with border replication
__global__ void resize_replicate_border_gpu(float *d_out, int pitch, int width,
                                            int height) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  *((float *)((char *)d_out + y *pitch) + x) = tex2D(imageTexture, x, y);
}

// integrated (non-separable) low-pass filtering and subsampling
// width and height refer to the subsampled imageTexture

__global__ void lpfSubsampleTexture(float *d_Out, int pitch, int width,
                                    int height) {
  const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x); // output
  const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y); // output
  const float x = 2.0f * (float)ix + 0.5f;                  // input
  const float y = 2.0f * (float)iy + 0.5f;                  // input

  if (ix >= width || iy >= height)
    return;

  float sum = 0.0f;
  sum += tex2D(imageTexture, x + 2.0f, y + 2.0f) * 0.0039062500000000f;
  sum += tex2D(imageTexture, x + 1.0f, y + 2.0f) * 0.0156250000000000f;
  sum += tex2D(imageTexture, x + 0.0f, y + 2.0f) * 0.0234375000000000f;
  sum += tex2D(imageTexture, x + -1.0f, y + 2.0f) * 0.0156250000000000f;
  sum += tex2D(imageTexture, x + -2.0f, y + 2.0f) * 0.0039062500000000f;
  sum += tex2D(imageTexture, x + 2.0f, y + 1.0f) * 0.0156250000000000f;
  sum += tex2D(imageTexture, x + 1.0f, y + 1.0f) * 0.0625000000000000f;
  sum += tex2D(imageTexture, x + 0.0f, y + 1.0f) * 0.0937500000000000f;
  sum += tex2D(imageTexture, x + -1.0f, y + 1.0f) * 0.0625000000000000f;
  sum += tex2D(imageTexture, x + -2.0f, y + 1.0f) * 0.0156250000000000f;
  sum += tex2D(imageTexture, x + 2.0f, y + 0.0f) * 0.0234375000000000f;
  sum += tex2D(imageTexture, x + 1.0f, y + 0.0f) * 0.0937500000000000f;
  sum += tex2D(imageTexture, x + 0.0f, y + 0.0f) * 0.1406250000000000f;
  sum += tex2D(imageTexture, x + -1.0f, y + 0.0f) * 0.0937500000000000f;
  sum += tex2D(imageTexture, x + -2.0f, y + 0.0f) * 0.0234375000000000f;
  sum += tex2D(imageTexture, x + 2.0f, y + -1.0f) * 0.0156250000000000f;
  sum += tex2D(imageTexture, x + 1.0f, y + -1.0f) * 0.0625000000000000f;
  sum += tex2D(imageTexture, x + 0.0f, y + -1.0f) * 0.0937500000000000f;
  sum += tex2D(imageTexture, x + -1.0f, y + -1.0f) * 0.0625000000000000f;
  sum += tex2D(imageTexture, x + -2.0f, y + -1.0f) * 0.0156250000000000f;
  sum += tex2D(imageTexture, x + 2.0f, y + -2.0f) * 0.0039062500000000f;
  sum += tex2D(imageTexture, x + 1.0f, y + -2.0f) * 0.0156250000000000f;
  sum += tex2D(imageTexture, x + 0.0f, y + -2.0f) * 0.0234375000000000f;
  sum += tex2D(imageTexture, x + -1.0f, y + -2.0f) * 0.0156250000000000f;
  sum += tex2D(imageTexture, x + -2.0f, y + -2.0f) * 0.0039062500000000f;

  *((float *)((char *)d_Out + iy *pitch) + ix) = sum;
}

// integrated (non-separable) low-pass filtering and subsampling
// width and height refer to the subsampled imageTexture
// this kernel deals with missing values

__global__ void lpfSubsampleTextureNaN(float *d_Out, int pitch, int width,
                                       int height) {
  const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x); // output
  const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y); // output
  const float x = 2.0f * (float)ix + 0.5f;                  // input
  const float y = 2.0f * (float)iy + 0.5f;                  // input

  if (ix >= width || iy >= height)
    return;

  float sum = 0.0f, summask = 0.0f;
  float data;

  data = tex2D(imageTexture, x + 2.0f, y + 2.0f);
  if (isfinite(data)) {
    summask += 0.0039062500000000f;
    sum += data * 0.0039062500000000f;
  }

  data = tex2D(imageTexture, x + 1.0f, y + 2.0f);
  if (isfinite(data)) {
    summask += 0.0156250000000000f;
    sum += data * 0.0156250000000000f;
  }
  data = tex2D(imageTexture, x + 0.0f, y + 2.0f);
  if (isfinite(data)) {
    summask += 0.0234375000000000f;
    sum += data * 0.0234375000000000f;
  }
  data = tex2D(imageTexture, x + -1.0f, y + 2.0f);
  if (isfinite(data)) {
    summask += 0.0156250000000000f;
    sum += data * 0.0156250000000000f;
  }
  data = tex2D(imageTexture, x + -2.0f, y + 2.0f);
  if (isfinite(data)) {
    summask += 0.0039062500000000f;
    sum += data * 0.0039062500000000f;
  }
  data = tex2D(imageTexture, x + 2.0f, y + 1.0f);
  if (isfinite(data)) {
    summask += 0.0156250000000000f;
    sum += data * 0.0156250000000000f;
  }
  data = tex2D(imageTexture, x + 1.0f, y + 1.0f);
  if (isfinite(data)) {
    summask += 0.0625000000000000f;
    sum += data * 0.0625000000000000f;
  }
  data = tex2D(imageTexture, x + 0.0f, y + 1.0f);
  if (isfinite(data)) {
    summask += 0.0937500000000000f;
    sum += data * 0.0937500000000000f;
  }
  data = tex2D(imageTexture, x + -1.0f, y + 1.0f);
  if (isfinite(data)) {
    summask += 0.0625000000000000f;
    sum += data * 0.0625000000000000f;
  }
  data = tex2D(imageTexture, x + -2.0f, y + 1.0f);
  if (isfinite(data)) {
    summask += 0.0156250000000000f;
    sum += data * 0.0156250000000000f;
  }
  data = tex2D(imageTexture, x + 2.0f, y + 0.0f);
  if (isfinite(data)) {
    summask += 0.0234375000000000f;
    sum += data * 0.0234375000000000f;
  }
  data = tex2D(imageTexture, x + 1.0f, y + 0.0f);
  if (isfinite(data)) {
    summask += 0.0937500000000000f;
    sum += data * 0.0937500000000000f;
  }
  data = tex2D(imageTexture, x + 0.0f, y + 0.0f);
  if (isfinite(data)) {
    summask += 0.1406250000000000f;
    sum += data * 0.1406250000000000f;
  }
  data = tex2D(imageTexture, x + -1.0f, y + 0.0f);
  if (isfinite(data)) {
    summask += 0.0937500000000000f;
    sum += data * 0.0937500000000000f;
  }
  data = tex2D(imageTexture, x + -2.0f, y + 0.0f);
  if (isfinite(data)) {
    summask += 0.0234375000000000f;
    sum += data * 0.0234375000000000f;
  }
  data = tex2D(imageTexture, x + 2.0f, y + -1.0f);
  if (isfinite(data)) {
    summask += 0.0156250000000000f;
    sum += data * 0.0156250000000000f;
  }
  data = tex2D(imageTexture, x + 1.0f, y + -1.0f);
  if (isfinite(data)) {
    summask += 0.0625000000000000f;
    sum += data * 0.0625000000000000f;
  }
  data = tex2D(imageTexture, x + 0.0f, y + -1.0f);
  if (isfinite(data)) {
    summask += 0.0937500000000000f;
    sum += data * 0.0937500000000000f;
  }
  data = tex2D(imageTexture, x + -1.0f, y + -1.0f);
  if (isfinite(data)) {
    summask += 0.0625000000000000f;
    sum += data * 0.0625000000000000f;
  }
  data = tex2D(imageTexture, x + -2.0f, y + -1.0f);
  if (isfinite(data)) {
    summask += 0.0156250000000000f;
    sum += data * 0.0156250000000000f;
  }
  data = tex2D(imageTexture, x + 2.0f, y + -2.0f);
  if (isfinite(data)) {
    summask += 0.0039062500000000f;
    sum += data * 0.0039062500000000f;
  }
  data = tex2D(imageTexture, x + 1.0f, y + -2.0f);
  if (isfinite(data)) {
    summask += 0.0156250000000000f;
    sum += data * 0.0156250000000000f;
  }
  data = tex2D(imageTexture, x + 0.0f, y + -2.0f);
  if (isfinite(data)) {
    summask += 0.0234375000000000f;
    sum += data * 0.0234375000000000f;
  }
  data = tex2D(imageTexture, x + -1.0f, y + -2.0f);
  if (isfinite(data)) {
    summask += 0.0156250000000000f;
    sum += data * 0.0156250000000000f;
  }
  data = tex2D(imageTexture, x + -2.0f, y + -2.0f);
  if (isfinite(data)) {
    summask += 0.0039062500000000f;
    sum += data * 0.0039062500000000f;
  }

  *((float *)((char *)d_Out + iy *pitch) + ix) = sum / summask;
}

__global__ void convolutionRowTexture(float2 *d_out, int width, int height,
                                      int pitch) {

  const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
  const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
  const float x = (float)ix + 0.5f;
  float y = (float)iy + 0.5f;

  if (ix >= width || iy >= height)
    return;

  //////////////////////////////////////////////// 0_1

  float pixel;
  float2 sum0 = make_float2(0.0f, 0.0f);

  pixel = tex2D(floatTexture, x + 5.0f, y);
  sum0.x += pixel * -0.0059423308933587f;
  sum0.y += pixel * -0.0854403050734955f;
  pixel = tex2D(floatTexture, x + 4.0f, y);
  sum0.x += pixel * 0.1996709851458967f;
  sum0.y += pixel * 0.0005340226360081f;
  pixel = tex2D(floatTexture, x + 3.0f, y);
  sum0.x += pixel * -0.0064730167337227f;
  sum0.y += pixel * 0.4124452895315693f;
  pixel = tex2D(floatTexture, x + 2.0f, y);
  sum0.x += pixel * -0.6814284613758747f;
  sum0.y += pixel * -0.0005666721140656f;
  pixel = tex2D(floatTexture, x + 1.0f, y);
  sum0.x += pixel * -0.0058271761429405f;
  sum0.y += pixel * -0.9093473751107518f;
  pixel = tex2D(floatTexture, x + 0.0f, y);
  sum0.x += pixel * 1.0000000000000000f;
  sum0.y += pixel * 0.0000000000000000f;
  pixel = tex2D(floatTexture, x + -1.0f, y);
  sum0.x += pixel * -0.0058271761429405f;
  sum0.y += pixel * 0.9093473751107518f;
  pixel = tex2D(floatTexture, x + -2.0f, y);
  sum0.x += pixel * -0.6814284613758747f;
  sum0.y += pixel * 0.0005666721140656f;
  pixel = tex2D(floatTexture, x + -3.0f, y);
  sum0.x += pixel * -0.0064730167337227f;
  sum0.y += pixel * -0.4124452895315693f;
  pixel = tex2D(floatTexture, x + -4.0f, y);
  sum0.x += pixel * 0.1996709851458967f;
  sum0.y += pixel * -0.0005340226360081f;
  pixel = tex2D(floatTexture, x + -5.0f, y);
  sum0.x += pixel * -0.0059423308933587f;
  sum0.y += pixel * 0.0854403050734955f;

  *((float2 *)((char *)d_out + iy *pitch) + ix) = sum0; // 0 even and odd

  //////////////////////////////////////////////// 0_2

  y += (float)height;

  float2 pixel2;
  sum0.x = 0.0f;
  sum0.y = 0.0f;

  pixel2 = tex2D(float2Texture, x + 5.0f, y);
  sum0.x += pixel2.x * 0.0824462481622174f;
  sum0.y += pixel2.y * 0.0824462481622174f;
  pixel2 = tex2D(float2Texture, x + 4.0f, y);
  sum0.x += pixel2.x * 0.2046904635506605f;
  sum0.y += pixel2.y * 0.2046904635506605f;
  pixel2 = tex2D(float2Texture, x + 3.0f, y);
  sum0.x += pixel2.x * 0.4107230390429492f;
  sum0.y += pixel2.y * 0.4107230390429492f;
  pixel2 = tex2D(float2Texture, x + 2.0f, y);
  sum0.x += pixel2.x * 0.6742558727374832f;
  sum0.y += pixel2.y * 0.6742558727374832f;
  pixel2 = tex2D(float2Texture, x + 1.0f, y);
  sum0.x += pixel2.x * 0.9070814926070788f;
  sum0.y += pixel2.y * 0.9070814926070788f;
  pixel2 = tex2D(float2Texture, x + 0.0f, y);
  sum0.x += pixel2.x * 0.9998288981228244f;
  sum0.y += pixel2.y * 0.9998288981228244f;
  pixel2 = tex2D(float2Texture, x + -1.0f, y);
  sum0.x += pixel2.x * 0.9070814926070788f;
  sum0.y += pixel2.y * 0.9070814926070788f;
  pixel2 = tex2D(float2Texture, x + -2.0f, y);
  sum0.x += pixel2.x * 0.6742558727374832f;
  sum0.y += pixel2.y * 0.6742558727374832f;
  pixel2 = tex2D(float2Texture, x + -3.0f, y);
  sum0.x += pixel2.x * 0.4107230390429492f;
  sum0.y += pixel2.y * 0.4107230390429492f;
  pixel2 = tex2D(float2Texture, x + -4.0f, y);
  sum0.x += pixel2.x * 0.2046904635506605f;
  sum0.y += pixel2.y * 0.2046904635506605f;
  pixel2 = tex2D(float2Texture, x + -5.0f, y);
  sum0.x += pixel2.x * 0.0824462481622174f;
  sum0.y += pixel2.y * 0.0824462481622174f;

  *((float2 *)((char *)d_out + (iy + 4 *height) *pitch) + ix) =
      sum0; // 90 even and odd

  //////////////////////////////////////////////// 1

  y += (float)height;
  sum0.x = 0.0f;
  sum0.y = 0.0f;
  float2 sum1 = make_float2(0.0f, 0.0f);

  pixel2 = tex2D(float2Texture, x + 5.0f, y);
  sum0.x += pixel2.x * 0.0448539697327717f;
  sum0.y += pixel2.x * 0.0563848093336092f;
  sum1.x += pixel2.y * 0.0563848093336092f;
  sum1.y += pixel2.y * 0.0448539697327717f;
  pixel2 = tex2D(float2Texture, x + 4.0f, y);
  sum0.x += pixel2.x * -0.0728676598565544f;
  sum0.y += pixel2.x * 0.1985363845521255f;
  sum1.x += pixel2.y * 0.1985363845521255f;
  sum1.y += pixel2.y * -0.0728676598565544f;
  pixel2 = tex2D(float2Texture, x + 3.0f, y);
  sum0.x += pixel2.x * -0.4218122479628296f;
  sum0.y += pixel2.x * 0.0779500055097176f;
  sum1.x += pixel2.y * 0.0779500055097176f;
  sum1.y += pixel2.y * -0.4218122479628296f;
  pixel2 = tex2D(float2Texture, x + 2.0f, y);
  sum0.x += pixel2.x * -0.4264028852345470f;
  sum0.y += pixel2.x * -0.5368628619030967f;
  sum1.x += pixel2.y * -0.5368628619030967f;
  sum1.y += pixel2.y * -0.4264028852345470f;
  pixel2 = tex2D(float2Texture, x + 1.0f, y);
  sum0.x += pixel2.x * 0.3845516108160854f;
  sum0.y += pixel2.x * -0.8133545314478231f;
  sum1.x += pixel2.y * -0.8133545314478231f;
  sum1.y += pixel2.y * 0.3845516108160854f;
  pixel2 = tex2D(float2Texture, x + 0.0f, y);
  sum0.x += pixel2.x * 0.9833544262984621f;
  sum0.y += pixel2.x * -0.0000000012323343f;
  sum1.x += pixel2.y * -0.0000000012323343f;
  sum1.y += pixel2.y * 0.9833544262984621f;
  pixel2 = tex2D(float2Texture, x + -1.0f, y);
  sum0.x += pixel2.x * 0.3845516108160854f;
  sum0.y += pixel2.x * 0.8133545314478231f;
  sum1.x += pixel2.y * 0.8133545314478231f;
  sum1.y += pixel2.y * 0.3845516108160854f;
  pixel2 = tex2D(float2Texture, x + -2.0f, y);
  sum0.x += pixel2.x * -0.4264028852345470f;
  sum0.y += pixel2.x * 0.5368628619030967f;
  sum1.x += pixel2.y * 0.5368628619030967f;
  sum1.y += pixel2.y * -0.4264028852345470f;
  pixel2 = tex2D(float2Texture, x + -3.0f, y);
  sum0.x += pixel2.x * -0.4218122479628296f;
  sum0.y += pixel2.x * -0.0779500055097176f;
  sum1.x += pixel2.y * -0.0779500055097176f;
  sum1.y += pixel2.y * -0.4218122479628296f;
  pixel2 = tex2D(float2Texture, x + -4.0f, y);
  sum0.x += pixel2.x * -0.0728676598565544f;
  sum0.y += pixel2.x * -0.1985363845521255f;
  sum1.x += pixel2.y * -0.1985363845521255f;
  sum1.y += pixel2.y * -0.0728676598565544f;
  pixel2 = tex2D(float2Texture, x + -5.0f, y);
  sum0.x += pixel2.x * 0.0448539697327717f;
  sum0.y += pixel2.x * -0.0563848093336092f;
  sum1.x += pixel2.y * -0.0563848093336092f;
  sum1.y += pixel2.y * 0.0448539697327717f;

  // combination stage

  pixel2.x = sum0.x - sum1.x; // 45 even F4YF4X-F5YF5X
  pixel2.y = sum1.y + sum0.y; // 45 odd  F5YF4X+F4YF5X
  *((float2 *)((char *)d_out + (iy + 2 *height) *pitch) + ix) = pixel2;

  pixel2.x = sum0.x + sum1.x; // 135 even F4YF4X+F5YF5X
  pixel2.y = sum1.y - sum0.y; // 135 odd  F5YF4X-F4YF5X
  *((float2 *)((char *)d_out + (iy + 6 *height) *pitch) + ix) = pixel2;

  //////////////////////////////////////////////// 2

  y += (float)height;
  sum0.x = 0.0f;
  sum0.y = 0.0f;
  sum1.x = 0.0f;
  sum1.y = 0.0f;

  pixel2 = tex2D(float2Texture, x + 5.0f, y);
  sum0.x += pixel2.x * -0.0865021727156619f;
  sum0.y += pixel2.x * -0.0082494019300884f;
  sum1.x += pixel2.y * -0.0082494019300884f;
  sum1.y += pixel2.y * -0.0865021727156619f;
  pixel2 = tex2D(float2Texture, x + 4.0f, y);
  sum0.x += pixel2.x * -0.1544706682064459f;
  sum0.y += pixel2.x * -0.1387420567977273f;
  sum1.x += pixel2.y * -0.1387420567977273f;
  sum1.y += pixel2.y * -0.1544706682064459f;
  pixel2 = tex2D(float2Texture, x + 3.0f, y);
  sum0.x += pixel2.x * -0.0961909886276083f;
  sum0.y += pixel2.x * -0.4004431484945309f;
  sum1.x += pixel2.y * -0.4004431484945309f;
  sum1.y += pixel2.y * -0.0961909886276083f;
  pixel2 = tex2D(float2Texture, x + 2.0f, y);
  sum0.x += pixel2.x * 0.2425229792248418f;
  sum0.y += pixel2.x * -0.6316382348347102f;
  sum1.x += pixel2.y * -0.6316382348347102f;
  sum1.y += pixel2.y * 0.2425229792248418f;
  pixel2 = tex2D(float2Texture, x + 1.0f, y);
  sum0.x += pixel2.x * 0.7444812173872333f;
  sum0.y += pixel2.x * -0.5161793771775458f;
  sum1.x += pixel2.y * -0.5161793771775458f;
  sum1.y += pixel2.y * 0.7444812173872333f;
  pixel2 = tex2D(float2Texture, x + 0.0f, y);
  sum0.x += pixel2.x * 0.9999674491845810f;
  sum0.y += pixel2.x * 0.0000034368466824f;
  sum1.x += pixel2.y * 0.0000034368466824f;
  sum1.y += pixel2.y * 0.9999674491845810f;
  pixel2 = tex2D(float2Texture, x + -1.0f, y);
  sum0.x += pixel2.x * 0.7444812173872333f;
  sum0.y += pixel2.x * 0.5161793771775458f;
  sum1.x += pixel2.y * 0.5161793771775458f;
  sum1.y += pixel2.y * 0.7444812173872333f;
  pixel2 = tex2D(float2Texture, x + -2.0f, y);
  sum0.x += pixel2.x * 0.2425229792248418f;
  sum0.y += pixel2.x * 0.6316382348347102f;
  sum1.x += pixel2.y * 0.6316382348347102f;
  sum1.y += pixel2.y * 0.2425229792248418f;
  pixel2 = tex2D(float2Texture, x + -3.0f, y);
  sum0.x += pixel2.x * -0.0961909886276083f;
  sum0.y += pixel2.x * 0.4004431484945309f;
  sum1.x += pixel2.y * 0.4004431484945309f;
  sum1.y += pixel2.y * -0.0961909886276083f;
  pixel2 = tex2D(float2Texture, x + -4.0f, y);
  sum0.x += pixel2.x * -0.1544706682064459f;
  sum0.y += pixel2.x * 0.1387420567977273f;
  sum1.x += pixel2.y * 0.1387420567977273f;
  sum1.y += pixel2.y * -0.1544706682064459f;
  pixel2 = tex2D(float2Texture, x + -5.0f, y);
  sum0.x += pixel2.x * -0.0865021727156619f;
  sum0.y += pixel2.x * 0.0082494019300884f;
  sum1.x += pixel2.y * 0.0082494019300884f;
  sum1.y += pixel2.y * -0.0865021727156619f;

  // combination stage

  pixel2.x = sum0.x - sum1.x; // 67.5 even F6YF8X-F7YF9X
  pixel2.y = sum1.y + sum0.y; // 67.5 odd  F7YF8X+F6YF9X
  *((float2 *)((char *)d_out + (iy + 3 *height) *pitch) + ix) = pixel2;

  pixel2.x = sum0.x + sum1.x; // 112.5 even F6YF8X+F7YF9X
  pixel2.y = sum1.y - sum0.y; // 112.5 odd  F7YF8X-F6YF9X
  *((float2 *)((char *)d_out + (iy + 5 *height) *pitch) + ix) = pixel2;

  //////////////////////////////////////////////// 3

  y += (float)height;
  sum0.x = 0.0f;
  sum0.y = 0.0f;
  sum1.x = 0.0f;
  sum1.y = 0.0f;

  pixel2 = tex2D(float2Texture, x + 5.0f, y);
  sum0.x += pixel2.x * 0.0422085900963304f;
  sum0.y += pixel2.x * -0.0702919919517587f;
  sum1.x += pixel2.y * -0.0702919919517587f;
  sum1.y += pixel2.y * 0.0422085900963304f;
  pixel2 = tex2D(float2Texture, x + 4.0f, y);
  sum0.x += pixel2.x * 0.1753604466059522f;
  sum0.y += pixel2.x * 0.0985418840494423f;
  sum1.x += pixel2.y * 0.0985418840494423f;
  sum1.y += pixel2.y * 0.1753604466059522f;
  pixel2 = tex2D(float2Texture, x + 3.0f, y);
  sum0.x += pixel2.x * -0.1499117207828439f;
  sum0.y += pixel2.x * 0.3900078591931931f;
  sum1.x += pixel2.y * 0.3900078591931931f;
  sum1.y += pixel2.y * -0.1499117207828439f;
  pixel2 = tex2D(float2Texture, x + 2.0f, y);
  sum0.x += pixel2.x * -0.6656505296765876f;
  sum0.y += pixel2.x * -0.1608071493187968f;
  sum1.x += pixel2.y * -0.1608071493187968f;
  sum1.y += pixel2.y * -0.6656505296765876f;
  pixel2 = tex2D(float2Texture, x + 1.0f, y);
  sum0.x += pixel2.x * 0.0996765973979726f;
  sum0.y += pixel2.x * -0.9011408273947247f;
  sum1.x += pixel2.y * -0.9011408273947247f;
  sum1.y += pixel2.y * 0.0996765973979726f;
  pixel2 = tex2D(float2Texture, x + 0.0f, y);
  sum0.x += pixel2.x * 0.9966332327183527f;
  sum0.y += pixel2.x * -0.0000000000000001f;
  sum1.x += pixel2.y * -0.0000000000000001f;
  sum1.y += pixel2.y * 0.9966332327183527f;
  pixel2 = tex2D(float2Texture, x + -1.0f, y);
  sum0.x += pixel2.x * 0.0996765973979726f;
  sum0.y += pixel2.x * 0.9011408273947247f;
  sum1.x += pixel2.y * 0.9011408273947247f;
  sum1.y += pixel2.y * 0.0996765973979726f;
  pixel2 = tex2D(float2Texture, x + -2.0f, y);
  sum0.x += pixel2.x * -0.6656505296765876f;
  sum0.y += pixel2.x * 0.1608071493187968f;
  sum1.x += pixel2.y * 0.1608071493187968f;
  sum1.y += pixel2.y * -0.6656505296765876f;
  pixel2 = tex2D(float2Texture, x + -3.0f, y);
  sum0.x += pixel2.x * -0.1499117207828439f;
  sum0.y += pixel2.x * -0.3900078591931931f;
  sum1.x += pixel2.y * -0.3900078591931931f;
  sum1.y += pixel2.y * -0.1499117207828439f;
  pixel2 = tex2D(float2Texture, x + -4.0f, y);
  sum0.x += pixel2.x * 0.1753604466059522f;
  sum0.y += pixel2.x * -0.0985418840494423f;
  sum1.x += pixel2.y * -0.0985418840494423f;
  sum1.y += pixel2.y * 0.1753604466059522f;
  pixel2 = tex2D(float2Texture, x + -5.0f, y);
  sum0.x += pixel2.x * 0.0422085900963304f;
  sum0.y += pixel2.x * 0.0702919919517587f;
  sum1.x += pixel2.y * 0.0702919919517587f;
  sum1.y += pixel2.y * 0.0422085900963304f;

  // combination stage

  pixel2.x = sum0.x - sum1.x; // 22.5 even F8YF6X-F9YF7X
  pixel2.y = sum1.y + sum0.y; // 22.5 odd  F9YF6X+F8YF7X
  *((float2 *)((char *)d_out + (iy + height) *pitch) + ix) = pixel2;

  pixel2.x = sum0.x + sum1.x; // 157.5 even F8YF6X+F9YF7X
  pixel2.y = sum1.y - sum0.y; // 157.5 odd  F9YF6X-F8YF7X
  *((float2 *)((char *)d_out + (iy + 7 *height) *pitch) + ix) = pixel2;
}

__global__ void convolutionColumnTexture(float *d_out, int width, int height,
                                         int pitch) {
  const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
  const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
  const float x = (float)ix + 0.5f;
  const float y = (float)iy + 0.5f;

  if (ix >= width || iy >= height)
    return;

  float pixel;
  float sum01 = 0.0f;
  float2 sum02 = make_float2(0.0f, 0.0f);
  float2 sum1 = make_float2(0.0f, 0.0f);
  float2 sum2 = make_float2(0.0f, 0.0f);
  float2 sum3 = make_float2(0.0f, 0.0f);

  pixel = tex2D(imageTexture, x, y + 5.0f);
  sum01 += pixel * 0.0824462481622174f;
  sum02.x += pixel * -0.0059423308933587f;
  sum02.y += pixel * -0.0854403050734955f;
  sum1.x += pixel * 0.0448539697327717f;
  sum1.y += pixel * 0.0563848093336092f;
  sum2.x += pixel * 0.0422085900963304f;
  sum2.y += pixel * -0.0702919919517587f;
  sum3.x += pixel * -0.0865021727156619f;
  sum3.y += pixel * -0.0082494019300884f;
  pixel = tex2D(imageTexture, x, y + 4.0f);
  sum01 += pixel * 0.2046904635506605f;
  sum02.x += pixel * 0.1996709851458967f;
  sum02.y += pixel * 0.0005340226360081f;
  sum1.x += pixel * -0.0728676598565544f;
  sum1.y += pixel * 0.1985363845521255f;
  sum2.x += pixel * 0.1753604466059522f;
  sum2.y += pixel * 0.0985418840494423f;
  sum3.x += pixel * -0.1544706682064459f;
  sum3.y += pixel * -0.1387420567977273f;
  pixel = tex2D(imageTexture, x, y + 3.0f);
  sum01 += pixel * 0.4107230390429492f;
  sum02.x += pixel * -0.0064730167337227f;
  sum02.y += pixel * 0.4124452895315693f;
  sum1.x += pixel * -0.4218122479628296f;
  sum1.y += pixel * 0.0779500055097176f;
  sum2.x += pixel * -0.1499117207828439f;
  sum2.y += pixel * 0.3900078591931931f;
  sum3.x += pixel * -0.0961909886276083f;
  sum3.y += pixel * -0.4004431484945309f;
  pixel = tex2D(imageTexture, x, y + 2.0f);
  sum01 += pixel * 0.6742558727374832f;
  sum02.x += pixel * -0.6814284613758747f;
  sum02.y += pixel * -0.0005666721140656f;
  sum1.x += pixel * -0.4264028852345470f;
  sum1.y += pixel * -0.5368628619030967f;
  sum2.x += pixel * -0.6656505296765876f;
  sum2.y += pixel * -0.1608071493187968f;
  sum3.x += pixel * 0.2425229792248418f;
  sum3.y += pixel * -0.6316382348347102f;
  pixel = tex2D(imageTexture, x, y + 1.0f);
  sum01 += pixel * 0.9070814926070788f;
  sum02.x += pixel * -0.0058271761429405f;
  sum02.y += pixel * -0.9093473751107518f;
  sum1.x += pixel * 0.3845516108160854f;
  sum1.y += pixel * -0.8133545314478231f;
  sum2.x += pixel * 0.0996765973979726f;
  sum2.y += pixel * -0.9011408273947247f;
  sum3.x += pixel * 0.7444812173872333f;
  sum3.y += pixel * -0.5161793771775458f;
  pixel = tex2D(imageTexture, x, y + 0.0f);
  sum01 += pixel * 0.9998288981228244f;
  sum02.x += pixel * 1.0000000000000000f;
  sum02.y += pixel * 0.0000000000000000f;
  sum1.x += pixel * 0.9833544262984621f;
  sum1.y += pixel * -0.0000000012323343f;
  sum2.x += pixel * 0.9966332327183527f;
  sum2.y += pixel * -0.0000000000000001f;
  sum3.x += pixel * 0.9999674491845810f;
  sum3.y += pixel * 0.0000034368466824f;
  pixel = tex2D(imageTexture, x, y + -1.0f);
  sum01 += pixel * 0.9070814926070788f;
  sum02.x += pixel * -0.0058271761429405f;
  sum02.y += pixel * 0.9093473751107518f;
  sum1.x += pixel * 0.3845516108160854f;
  sum1.y += pixel * 0.8133545314478231f;
  sum2.x += pixel * 0.0996765973979726f;
  sum2.y += pixel * 0.9011408273947247f;
  sum3.x += pixel * 0.7444812173872333f;
  sum3.y += pixel * 0.5161793771775458f;
  pixel = tex2D(imageTexture, x, y + -2.0f);
  sum01 += pixel * 0.6742558727374832f;
  sum02.x += pixel * -0.6814284613758747f;
  sum02.y += pixel * 0.0005666721140656f;
  sum1.x += pixel * -0.4264028852345470f;
  sum1.y += pixel * 0.5368628619030967f;
  sum2.x += pixel * -0.6656505296765876f;
  sum2.y += pixel * 0.1608071493187968f;
  sum3.x += pixel * 0.2425229792248418f;
  sum3.y += pixel * 0.6316382348347102f;
  pixel = tex2D(imageTexture, x, y + -3.0f);
  sum01 += pixel * 0.4107230390429492f;
  sum02.x += pixel * -0.0064730167337227f;
  sum02.y += pixel * -0.4124452895315693f;
  sum1.x += pixel * -0.4218122479628296f;
  sum1.y += pixel * -0.0779500055097176f;
  sum2.x += pixel * -0.1499117207828439f;
  sum2.y += pixel * -0.3900078591931931f;
  sum3.x += pixel * -0.0961909886276083f;
  sum3.y += pixel * 0.4004431484945309f;
  pixel = tex2D(imageTexture, x, y + -4.0f);
  sum01 += pixel * 0.2046904635506605f;
  sum02.x += pixel * 0.1996709851458967f;
  sum02.y += pixel * -0.0005340226360081f;
  sum1.x += pixel * -0.0728676598565544f;
  sum1.y += pixel * -0.1985363845521255f;
  sum2.x += pixel * 0.1753604466059522f;
  sum2.y += pixel * -0.0985418840494423f;
  sum3.x += pixel * -0.1544706682064459f;
  sum3.y += pixel * 0.1387420567977273f;
  pixel = tex2D(imageTexture, x, y + -5.0f);
  sum01 += pixel * 0.0824462481622174f;
  sum02.x += pixel * -0.0059423308933587f;
  sum02.y += pixel * 0.0854403050734955f;
  sum1.x += pixel * 0.0448539697327717f;
  sum1.y += pixel * -0.0563848093336092f;
  sum2.x += pixel * 0.0422085900963304f;
  sum2.y += pixel * 0.0702919919517587f;
  sum3.x += pixel * -0.0865021727156619f;
  sum3.y += pixel * 0.0082494019300884f;

  *((float *)((char *)d_out + iy *pitch) + ix) = sum01;
  *((float2 *)((char *)d_out + (iy + height) *pitch) + ix) = sum02;
  *((float2 *)((char *)d_out + (iy + 2 *height) *pitch) + ix) = sum1;
  *((float2 *)((char *)d_out + (iy + 3 *height) *pitch) + ix) = sum2;
  *((float2 *)((char *)d_out + (iy + 4 *height) *pitch) + ix) = sum3;
}

__global__ void convolutionRowTexture4(float2 *d_out, int width, int height,
                                       int pitch) {

  const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
  const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
  const float x = (float)ix + 0.5f;
  float y = (float)iy + 0.5f;

  if (ix >= width || iy >= height)
    return;

  //////////////////////////////////////////////// 0_1

  float pixel;
  float2 sum0 = make_float2(0.0f, 0.0f);

  pixel = tex2D(floatTexture, x + 5.0f, y);
  sum0.x += pixel * -0.0059423308933587f;
  sum0.y += pixel * -0.0854403050734955f;
  pixel = tex2D(floatTexture, x + 4.0f, y);
  sum0.x += pixel * 0.1996709851458967f;
  sum0.y += pixel * 0.0005340226360081f;
  pixel = tex2D(floatTexture, x + 3.0f, y);
  sum0.x += pixel * -0.0064730167337227f;
  sum0.y += pixel * 0.4124452895315693f;
  pixel = tex2D(floatTexture, x + 2.0f, y);
  sum0.x += pixel * -0.6814284613758747f;
  sum0.y += pixel * -0.0005666721140656f;
  pixel = tex2D(floatTexture, x + 1.0f, y);
  sum0.x += pixel * -0.0058271761429405f;
  sum0.y += pixel * -0.9093473751107518f;
  pixel = tex2D(floatTexture, x + 0.0f, y);
  sum0.x += pixel * 1.0000000000000000f;
  sum0.y += pixel * 0.0000000000000000f;
  pixel = tex2D(floatTexture, x + -1.0f, y);
  sum0.x += pixel * -0.0058271761429405f;
  sum0.y += pixel * 0.9093473751107518f;
  pixel = tex2D(floatTexture, x + -2.0f, y);
  sum0.x += pixel * -0.6814284613758747f;
  sum0.y += pixel * 0.0005666721140656f;
  pixel = tex2D(floatTexture, x + -3.0f, y);
  sum0.x += pixel * -0.0064730167337227f;
  sum0.y += pixel * -0.4124452895315693f;
  pixel = tex2D(floatTexture, x + -4.0f, y);
  sum0.x += pixel * 0.1996709851458967f;
  sum0.y += pixel * -0.0005340226360081f;
  pixel = tex2D(floatTexture, x + -5.0f, y);
  sum0.x += pixel * -0.0059423308933587f;
  sum0.y += pixel * 0.0854403050734955f;

  *((float2 *)((char *)d_out + iy *pitch) + ix) = sum0; // 0 even and odd

  //////////////////////////////////////////////// 0_2

  y += (float)height;

  float2 pixel2;
  sum0.x = 0.0f;
  sum0.y = 0.0f;

  pixel2 = tex2D(float2Texture, x + 5.0f, y);
  sum0.x += pixel2.x * 0.0824462481622174f;
  sum0.y += pixel2.y * 0.0824462481622174f;
  pixel2 = tex2D(float2Texture, x + 4.0f, y);
  sum0.x += pixel2.x * 0.2046904635506605f;
  sum0.y += pixel2.y * 0.2046904635506605f;
  pixel2 = tex2D(float2Texture, x + 3.0f, y);
  sum0.x += pixel2.x * 0.4107230390429492f;
  sum0.y += pixel2.y * 0.4107230390429492f;
  pixel2 = tex2D(float2Texture, x + 2.0f, y);
  sum0.x += pixel2.x * 0.6742558727374832f;
  sum0.y += pixel2.y * 0.6742558727374832f;
  pixel2 = tex2D(float2Texture, x + 1.0f, y);
  sum0.x += pixel2.x * 0.9070814926070788f;
  sum0.y += pixel2.y * 0.9070814926070788f;
  pixel2 = tex2D(float2Texture, x + 0.0f, y);
  sum0.x += pixel2.x * 0.9998288981228244f;
  sum0.y += pixel2.y * 0.9998288981228244f;
  pixel2 = tex2D(float2Texture, x + -1.0f, y);
  sum0.x += pixel2.x * 0.9070814926070788f;
  sum0.y += pixel2.y * 0.9070814926070788f;
  pixel2 = tex2D(float2Texture, x + -2.0f, y);
  sum0.x += pixel2.x * 0.6742558727374832f;
  sum0.y += pixel2.y * 0.6742558727374832f;
  pixel2 = tex2D(float2Texture, x + -3.0f, y);
  sum0.x += pixel2.x * 0.4107230390429492f;
  sum0.y += pixel2.y * 0.4107230390429492f;
  pixel2 = tex2D(float2Texture, x + -4.0f, y);
  sum0.x += pixel2.x * 0.2046904635506605f;
  sum0.y += pixel2.y * 0.2046904635506605f;
  pixel2 = tex2D(float2Texture, x + -5.0f, y);
  sum0.x += pixel2.x * 0.0824462481622174f;
  sum0.y += pixel2.y * 0.0824462481622174f;

  *((float2 *)((char *)d_out + (iy + 2 *height) *pitch) + ix) =
      sum0; // 90 even and odd

  //////////////////////////////////////////////// 1

  y += (float)height;
  sum0.x = 0.0f;
  sum0.y = 0.0f;
  float2 sum1 = make_float2(0.0f, 0.0f);

  pixel2 = tex2D(float2Texture, x + 5.0f, y);
  sum0.x += pixel2.x * 0.0448539697327717f;
  sum0.y += pixel2.x * 0.0563848093336092f;
  sum1.x += pixel2.y * 0.0563848093336092f;
  sum1.y += pixel2.y * 0.0448539697327717f;
  pixel2 = tex2D(float2Texture, x + 4.0f, y);
  sum0.x += pixel2.x * -0.0728676598565544f;
  sum0.y += pixel2.x * 0.1985363845521255f;
  sum1.x += pixel2.y * 0.1985363845521255f;
  sum1.y += pixel2.y * -0.0728676598565544f;
  pixel2 = tex2D(float2Texture, x + 3.0f, y);
  sum0.x += pixel2.x * -0.4218122479628296f;
  sum0.y += pixel2.x * 0.0779500055097176f;
  sum1.x += pixel2.y * 0.0779500055097176f;
  sum1.y += pixel2.y * -0.4218122479628296f;
  pixel2 = tex2D(float2Texture, x + 2.0f, y);
  sum0.x += pixel2.x * -0.4264028852345470f;
  sum0.y += pixel2.x * -0.5368628619030967f;
  sum1.x += pixel2.y * -0.5368628619030967f;
  sum1.y += pixel2.y * -0.4264028852345470f;
  pixel2 = tex2D(float2Texture, x + 1.0f, y);
  sum0.x += pixel2.x * 0.3845516108160854f;
  sum0.y += pixel2.x * -0.8133545314478231f;
  sum1.x += pixel2.y * -0.8133545314478231f;
  sum1.y += pixel2.y * 0.3845516108160854f;
  pixel2 = tex2D(float2Texture, x + 0.0f, y);
  sum0.x += pixel2.x * 0.9833544262984621f;
  sum0.y += pixel2.x * -0.0000000012323343f;
  sum1.x += pixel2.y * -0.0000000012323343f;
  sum1.y += pixel2.y * 0.9833544262984621f;
  pixel2 = tex2D(float2Texture, x + -1.0f, y);
  sum0.x += pixel2.x * 0.3845516108160854f;
  sum0.y += pixel2.x * 0.8133545314478231f;
  sum1.x += pixel2.y * 0.8133545314478231f;
  sum1.y += pixel2.y * 0.3845516108160854f;
  pixel2 = tex2D(float2Texture, x + -2.0f, y);
  sum0.x += pixel2.x * -0.4264028852345470f;
  sum0.y += pixel2.x * 0.5368628619030967f;
  sum1.x += pixel2.y * 0.5368628619030967f;
  sum1.y += pixel2.y * -0.4264028852345470f;
  pixel2 = tex2D(float2Texture, x + -3.0f, y);
  sum0.x += pixel2.x * -0.4218122479628296f;
  sum0.y += pixel2.x * -0.0779500055097176f;
  sum1.x += pixel2.y * -0.0779500055097176f;
  sum1.y += pixel2.y * -0.4218122479628296f;
  pixel2 = tex2D(float2Texture, x + -4.0f, y);
  sum0.x += pixel2.x * -0.0728676598565544f;
  sum0.y += pixel2.x * -0.1985363845521255f;
  sum1.x += pixel2.y * -0.1985363845521255f;
  sum1.y += pixel2.y * -0.0728676598565544f;
  pixel2 = tex2D(float2Texture, x + -5.0f, y);
  sum0.x += pixel2.x * 0.0448539697327717f;
  sum0.y += pixel2.x * -0.0563848093336092f;
  sum1.x += pixel2.y * -0.0563848093336092f;
  sum1.y += pixel2.y * 0.0448539697327717f;

  // combination stage

  pixel2.x = sum0.x - sum1.x; // 45 even F4YF4X-F5YF5X
  pixel2.y = sum1.y + sum0.y; // 45 odd  F5YF4X+F4YF5X
  *((float2 *)((char *)d_out + (iy + 1 *height) *pitch) + ix) = pixel2;

  pixel2.x = sum0.x + sum1.x; // 135 even F4YF4X+F5YF5X
  pixel2.y = sum1.y - sum0.y; // 135 odd  F5YF4X-F4YF5X
  *((float2 *)((char *)d_out + (iy + 3 *height) *pitch) + ix) = pixel2;
}

__global__ void convolutionColumnTexture4(float *d_out, int width, int height,
                                          int pitch) {
  const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
  const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
  const float x = (float)ix + 0.5f;
  const float y = (float)iy + 0.5f;

  if (ix >= width || iy >= height)
    return;

  float pixel;
  float sum01 = 0.0f;
  float2 sum02 = make_float2(0.0f, 0.0f);
  float2 sum1 = make_float2(0.0f, 0.0f);

  pixel = tex2D(imageTexture, x, y + 5.0f);
  sum01 += pixel * 0.0824462481622174f;
  sum02.x += pixel * -0.0059423308933587f;
  sum02.y += pixel * -0.0854403050734955f;
  sum1.x += pixel * 0.0448539697327717f;
  sum1.y += pixel * 0.0563848093336092f;
  pixel = tex2D(imageTexture, x, y + 4.0f);
  sum01 += pixel * 0.2046904635506605f;
  sum02.x += pixel * 0.1996709851458967f;
  sum02.y += pixel * 0.0005340226360081f;
  sum1.x += pixel * -0.0728676598565544f;
  sum1.y += pixel * 0.1985363845521255f;
  pixel = tex2D(imageTexture, x, y + 3.0f);
  sum01 += pixel * 0.4107230390429492f;
  sum02.x += pixel * -0.0064730167337227f;
  sum02.y += pixel * 0.4124452895315693f;
  sum1.x += pixel * -0.4218122479628296f;
  sum1.y += pixel * 0.0779500055097176f;
  pixel = tex2D(imageTexture, x, y + 2.0f);
  sum01 += pixel * 0.6742558727374832f;
  sum02.x += pixel * -0.6814284613758747f;
  sum02.y += pixel * -0.0005666721140656f;
  sum1.x += pixel * -0.4264028852345470f;
  sum1.y += pixel * -0.5368628619030967f;
  pixel = tex2D(imageTexture, x, y + 1.0f);
  sum01 += pixel * 0.9070814926070788f;
  sum02.x += pixel * -0.0058271761429405f;
  sum02.y += pixel * -0.9093473751107518f;
  sum1.x += pixel * 0.3845516108160854f;
  sum1.y += pixel * -0.8133545314478231f;
  pixel = tex2D(imageTexture, x, y + 0.0f);
  sum01 += pixel * 0.9998288981228244f;
  sum02.x += pixel * 1.0000000000000000f;
  sum02.y += pixel * 0.0000000000000000f;
  sum1.x += pixel * 0.9833544262984621f;
  sum1.y += pixel * -0.0000000012323343f;
  pixel = tex2D(imageTexture, x, y + -1.0f);
  sum01 += pixel * 0.9070814926070788f;
  sum02.x += pixel * -0.0058271761429405f;
  sum02.y += pixel * 0.9093473751107518f;
  sum1.x += pixel * 0.3845516108160854f;
  sum1.y += pixel * 0.8133545314478231f;
  pixel = tex2D(imageTexture, x, y + -2.0f);
  sum01 += pixel * 0.6742558727374832f;
  sum02.x += pixel * -0.6814284613758747f;
  sum02.y += pixel * 0.0005666721140656f;
  sum1.x += pixel * -0.4264028852345470f;
  sum1.y += pixel * 0.5368628619030967f;
  pixel = tex2D(imageTexture, x, y + -3.0f);
  sum01 += pixel * 0.4107230390429492f;
  sum02.x += pixel * -0.0064730167337227f;
  sum02.y += pixel * -0.4124452895315693f;
  sum1.x += pixel * -0.4218122479628296f;
  sum1.y += pixel * -0.0779500055097176f;
  pixel = tex2D(imageTexture, x, y + -4.0f);
  sum01 += pixel * 0.2046904635506605f;
  sum02.x += pixel * 0.1996709851458967f;
  sum02.y += pixel * -0.0005340226360081f;
  sum1.x += pixel * -0.0728676598565544f;
  sum1.y += pixel * -0.1985363845521255f;
  pixel = tex2D(imageTexture, x, y + -5.0f);
  sum01 += pixel * 0.0824462481622174f;
  sum02.x += pixel * -0.0059423308933587f;
  sum02.y += pixel * 0.0854403050734955f;
  sum1.x += pixel * 0.0448539697327717f;
  sum1.y += pixel * -0.0563848093336092f;

  *((float *)((char *)d_out + iy *pitch) + ix) = sum01;
  *((float2 *)((char *)d_out + (iy + height) *pitch) + ix) = sum02;
  *((float2 *)((char *)d_out + (iy + 2 *height) *pitch) + ix) = sum1;
}

//////////////////////////////////////////////////////////////
// Calling functions
///////////////////////////////////////////////////////////////

void resize_replicate_border(const float *d_PixelsIn, int d_PixelsInPitch,
                             float *d_PixelsOut, int d_PixelsOutPitch,
                             int width_in, int height_in, int width_out,
                             int height_out) {
  dim3 threads(16, 8);
  dim3 blocks(iDivUp(width_out, threads.x), iDivUp(height_out, threads.y));

  imageTexture.normalized = 0;
  imageTexture.filterMode = cudaFilterModePoint;
  imageTexture.addressMode[0] = cudaAddressModeClamp;
  imageTexture.addressMode[1] = cudaAddressModeClamp;
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();

  cudaBindTexture2D(0, &imageTexture, d_PixelsIn, &channelFloat, width_in,
                    height_in, d_PixelsInPitch);

  resize_replicate_border_gpu << <blocks, threads>>>
      (d_PixelsOut, d_PixelsOutPitch, width_out, height_out);

  cudaUnbindTexture(imageTexture);
}

void downSample(const float *d_PixelsIn, int d_PixelsInPitch,
                float *d_PixelsOut, int d_PixelsOutPitch, int width,
                int height) {
  int width_out = width / 2;
  int height_out = height / 2;
  dim3 threads(16, 8);
  dim3 blocks(iDivUp(width_out, threads.x), iDivUp(height_out, threads.y));

  imageTexture.normalized = 0;
  imageTexture.filterMode = cudaFilterModePoint;
  imageTexture.addressMode[0] = cudaAddressModeClamp;
  imageTexture.addressMode[1] = cudaAddressModeClamp;
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();

  cudaBindTexture2D(0, &imageTexture, d_PixelsIn, &channelFloat, width, height,
                    d_PixelsInPitch);

  lpfSubsampleTexture << <blocks, threads>>>
      (d_PixelsOut, d_PixelsOutPitch, width_out, height_out);

  cudaUnbindTexture(imageTexture);
}

void gaborFilterItl(const float *d_Image, int d_ImagePitch, float2 *d_GabItl,
                    int d_GabItlPitch, char *d_TEMP, int d_TEMPPitch, int width,
                    int height, bool fourOrientations) {

  //  printf("%p %d %p %d %p %d %d %d %d\n",d_Image, d_ImagePitch, d_GabItl,
  // d_GabItlPitch, d_TEMP, d_TEMPPitch, width, height, fourOrientations);

  // setup execution parameters

  dim3 threads(16, 8);
  dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));

  imageTexture.normalized = 0;
  imageTexture.filterMode = cudaFilterModePoint;
  imageTexture.addressMode[0] = cudaAddressModeClamp;
  imageTexture.addressMode[1] = cudaAddressModeClamp;
  floatTexture.normalized = 0;
  floatTexture.filterMode = cudaFilterModePoint;
  floatTexture.addressMode[0] = cudaAddressModeClamp;
  floatTexture.addressMode[1] = cudaAddressModeClamp;
  float2Texture.normalized = 0;
  float2Texture.filterMode = cudaFilterModePoint;
  float2Texture.addressMode[0] = cudaAddressModeClamp;
  float2Texture.addressMode[1] = cudaAddressModeClamp;

  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();
  cudaChannelFormatDesc channelFloat2 = cudaCreateChannelDesc<float2>();

  cudaBindTexture2D(0, &imageTexture, d_Image, &channelFloat, width, height,
                    d_ImagePitch);

  if (fourOrientations) {

    convolutionColumnTexture4 << <blocks, threads>>>
        ((float *)d_TEMP, width, height, d_TEMPPitch);
    cudaUnbindTexture(imageTexture);

    cudaBindTexture2D(0, &floatTexture, d_TEMP, &channelFloat, width, height,
                      d_TEMPPitch);
    cudaBindTexture2D(0, &float2Texture, d_TEMP, &channelFloat2, width,
                      3 * height, d_TEMPPitch);

    // the use of this big texture with the responses tiled on top of each other
    // is fine here since we do column filtering first, in the row filter we
    // only leave the image on the left and right side

    convolutionRowTexture4 << <blocks, threads>>>
        (d_GabItl, width, height, d_GabItlPitch);

  } else {

    convolutionColumnTexture << <blocks, threads>>>
        ((float *)d_TEMP, width, height, d_TEMPPitch);
    cudaUnbindTexture(imageTexture);

    cudaBindTexture2D(0, &floatTexture, d_TEMP, &channelFloat, width, height,
                      d_TEMPPitch);
    cudaBindTexture2D(0, &float2Texture, d_TEMP, &channelFloat2, width,
                      5 * height, d_TEMPPitch);

    // the use of this big texture with the responses tiled on top of each other
    // is fine here since we do column filtering first, in the row filter we
    // only leave the image on the left and right side

    convolutionRowTexture << <blocks, threads>>>
        (d_GabItl, width, height, d_GabItlPitch);
  }

  cudaUnbindTexture(floatTexture);
  cudaUnbindTexture(float2Texture);
}

void downSampleNaN(const float *d_PixelsIn, int d_PixelsInPitch,
                   float *d_PixelsOut, int d_PixelsOutPitch, int width,
                   int height) {
  int width_out = width / 2;
  int height_out = height / 2;
  dim3 threads(16, 8);
  dim3 blocks(iDivUp(width_out, threads.x), iDivUp(height_out, threads.y));

  imageTexture.normalized = 0;
  imageTexture.filterMode = cudaFilterModePoint;
  imageTexture.addressMode[0] = cudaAddressModeClamp;
  imageTexture.addressMode[1] = cudaAddressModeClamp;
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();

  cudaBindTexture2D(0, &imageTexture, d_PixelsIn, &channelFloat, width, height,
                    d_PixelsInPitch);

  lpfSubsampleTextureNaN << <blocks, threads>>>
      (d_PixelsOut, d_PixelsOutPitch, width_out, height_out);

  cudaUnbindTexture(imageTexture);
}

} // end namespace vision
