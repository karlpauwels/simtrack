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

//#define UNROLL_INNER
//#define IMUL(a, b) __mul24(a, b)
#include <utility_kernels.h>

namespace vision {

// OpenGL mapped input textures
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> d_rgba_texture;
texture<float, cudaTextureType2D, cudaReadModeElementType> d_float_texture0;
texture<float, cudaTextureType2D, cudaReadModeElementType> d_float_texture1;

// Round a / b to nearest higher integer value
int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

__device__ static float rgbaToGray(uchar4 rgba) {
  return (0.299f * (float)rgba.x + 0.587f * (float)rgba.y +
          0.114f * (float)rgba.z);
}

__global__ void deInterleave_kernel(float *d_X_out, float *d_Y_out,
                                    float2 *d_XY_in, int pitch_out,
                                    int pitch_in, int width, int height) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < width) & (y < height)) { // are we in the image?
    float2 XY = *((float2 *)((char *)d_XY_in + y * pitch_in) + x);
    *((float *)((char *)d_X_out + y *pitch_out) + x) = XY.x;
    *((float *)((char *)d_Y_out + y *pitch_out) + x) = XY.y;
  }
}

__global__ void deInterleave_kernel2(float *d_X_out, float *d_Y_out,
                                     char *d_XY_in, int pitch_out, int pitch_in,
                                     int width, int height) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < width) & (y < height)) { // are we in the image?
    float *data = (float *)(d_XY_in + y * pitch_in) + 2 * x;
    *((float *)((char *)d_X_out + y *pitch_out) + x) = data[0];
    *((float *)((char *)d_Y_out + y *pitch_out) + x) = data[1];
  }
}

__global__ void IMOMask_kernel(float *d_IMOMask, float *d_IMO,
                               const float *d_disparity, float offset,
                               int n_cols, int n_rows) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < n_cols) & (y < n_rows)) // are we in the image?
  {
    unsigned int ind = x + y * n_cols;
    if (!(bool)(d_IMOMask[ind])) {
      d_IMO[ind] = nanf("");
    }
  }
}

__global__ void matchValidity_kernel(float *d_flow, float *d_disparity,
                                     int n_cols, int n_rows) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < n_cols) & (y < n_rows)) // are we in the image?
  {
    unsigned int ind = x + y * n_cols;
    bool valid = (isfinite(d_flow[ind]) && isfinite(d_disparity[ind]));
    if (!valid) {
      d_flow[ind] = nanf("");
      d_flow[ind + n_cols * n_rows] = nanf("");
      d_disparity[ind] = nanf("");
    }
  }
}

// Convert float to RGBA kernel

#define GAIN (1.0f / STEREO_MAXD)

__global__ void convertFloatToRGBA_kernel(uchar4 *out_image,
                                          const float *in_image, int width,
                                          int height, float lowerLim,
                                          float upperLim) {
  const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
  uchar4 temp;
  if (x < width && y < height) {
    float val = in_image[__mul24(y, width) + x];

    // first draw unmatched pixels in white
    if (!isfinite(val)) {
      temp.x = 255;
      temp.y = 255;
      temp.z = 255;
      temp.w = 255;
    } else {
      // rescale value from [lowerLim,upperLim] to [0,1]
      val -= lowerLim;
      val /= (upperLim - lowerLim);

      float r = 1.0f;
      float g = 1.0f;
      float b = 1.0f;
      if (val < 0.25f) {
        r = 0;
        g = 4.0f * val;
      } else if (val < 0.5f) {
        r = 0;
        b = 1.0 + 4.0f * (0.25f - val);
      } else if (val < 0.75f) {
        r = 4.0f * (val - 0.5f);
        b = 0;
      } else {
        g = 1.0f + 4.0f * (0.75f - val);
        b = 0;
      }
      temp.x = 255.0 * r;
      temp.y = 255.0 * g;
      temp.z = 255.0 * b;
      temp.w = 255;
    }
    out_image[__mul24(y, width) + x] = temp;
  }
}

__global__ void convertPitchedFloatToRGBA_kernel(uchar4 *out_image,
                                                 const float *in_image,
                                                 int width, int height,
                                                 int pitch, float lowerLim,
                                                 float upperLim) {
  const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
  uchar4 temp;
  if (x < width && y < height) {
    float val = *((float *)((char *)in_image + y * pitch) + x);

    // first draw unmatched pixels in white
    if (!isfinite(val)) {
      temp.x = 255;
      temp.y = 255;
      temp.z = 255;
      temp.w = 255;
    } else {
      // rescale value from [lowerLim,upperLim] to [0,1]
      val -= lowerLim;
      val /= (upperLim - lowerLim);

      float r = 1.0f;
      float g = 1.0f;
      float b = 1.0f;
      if (val < 0.25f) {
        r = 0;
        g = 4.0f * val;
      } else if (val < 0.5f) {
        r = 0;
        b = 1.0 + 4.0f * (0.25f - val);
      } else if (val < 0.75f) {
        r = 4.0f * (val - 0.5f);
        b = 0;
      } else {
        g = 1.0f + 4.0f * (0.75f - val);
        b = 0;
      }
      temp.x = 255.0 * r;
      temp.y = 255.0 * g;
      temp.z = 255.0 * b;
      temp.w = 255;
    }
    out_image[__mul24(y, width) + x] = temp;
  }
}

__global__ void convertKinectFloatToRGBA_kernel(uchar4 *out_image,
                                                const float *in_image,
                                                int width, int height,
                                                int pitch, float lowerLim,
                                                float upperLim) {
  const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
  uchar4 temp;
  if (x < width && y < height) {
    float val = *((float *)((char *)in_image + y * pitch) + x);

    val = (val == 0.0f) ? nanf("") : val;

    // first draw unmatched pixels in white
    if (!isfinite(val)) {
      temp.x = 255;
      temp.y = 255;
      temp.z = 255;
      temp.w = 255;
    } else {
      // rescale value from [lowerLim,upperLim] to [0,1]
      val -= lowerLim;
      val /= (upperLim - lowerLim);

      float r = 1.0f;
      float g = 1.0f;
      float b = 1.0f;
      if (val < 0.25f) {
        r = 0;
        g = 4.0f * val;
      } else if (val < 0.5f) {
        r = 0;
        b = 1.0 + 4.0f * (0.25f - val);
      } else if (val < 0.75f) {
        r = 4.0f * (val - 0.5f);
        b = 0;
      } else {
        g = 1.0f + 4.0f * (0.75f - val);
        b = 0;
      }
      temp.x = 255.0 * r;
      temp.y = 255.0 * g;
      temp.z = 255.0 * b;
      temp.w = 255;
    }
    out_image[__mul24(y, width) + x] = temp;
  }
}

__global__ void convertFloatToRGBA_kernel(uchar4 *out_image,
                                          const float *in_image, int width,
                                          int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  uchar4 temp;
  if (x < width && y < height) {
    int IND = y * width + x;
    float val = in_image[IND];
    temp.x = val;
    temp.y = val;
    temp.z = val;
    temp.w = 255;
    out_image[IND] = temp;
  }
}

__global__ void convertFlowToRGBA_kernel(uchar4 *d_flowx_out,
                                         uchar4 *d_flowy_out,
                                         const float *d_flowx_in,
                                         const float *d_flowy_in, int width,
                                         int height, float lowerLim,
                                         float upperLim, float minMag) {
  const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
  uchar4 tempx, tempy;
  if (x < width && y < height) {
    float ux = d_flowx_in[__mul24(y, width) + x];
    float uy = d_flowy_in[__mul24(y, width) + x];

    float mag = sqrtf(ux * ux + uy * uy);

    // first draw unmatched pixels in white
    if (!isfinite(ux) || (mag < minMag)) {

      tempx.x = 255;
      tempx.y = 255;
      tempx.z = 255;
      tempx.w = 255;
      tempy.x = 255;
      tempy.y = 255;
      tempy.z = 255;
      tempy.w = 255;

    } else {

      // rescale value from [lowerLim,upperLim] to [0,1]
      ux -= lowerLim;
      ux /= (upperLim - lowerLim);

      float r = 1.0f;
      float g = 1.0f;
      float b = 1.0f;
      if (ux < 0.25f) {
        r = 0;
        g = 4.0f * ux;
      } else if (ux < 0.5f) {
        r = 0;
        b = 1.0 + 4.0f * (0.25f - ux);
      } else if (ux < 0.75f) {
        r = 4.0f * (ux - 0.5f);
        b = 0;
      } else {
        g = 1.0f + 4.0f * (0.75f - ux);
        b = 0;
      }
      tempx.x = 255.0 * r;
      tempx.y = 255.0 * g;
      tempx.z = 255.0 * b;
      tempx.w = 255;

      uy -= lowerLim;
      uy /= (upperLim - lowerLim);

      r = 1.0f;
      g = 1.0f;
      b = 1.0f;
      if (uy < 0.25f) {
        r = 0;
        g = 4.0f * uy;
      } else if (uy < 0.5f) {
        r = 0;
        b = 1.0 + 4.0f * (0.25f - uy);
      } else if (uy < 0.75f) {
        r = 4.0f * (uy - 0.5f);
        b = 0;
      } else {
        g = 1.0f + 4.0f * (0.75f - uy);
        b = 0;
      }
      tempy.x = 255.0 * r;
      tempy.y = 255.0 * g;
      tempy.z = 255.0 * b;
      tempy.w = 255;
    }

    d_flowx_out[__mul24(y, width) + x] = tempx;
    d_flowy_out[__mul24(y, width) + x] = tempy;
  }
}

// Convert pitched float to RGBA grayscale
__global__ void convertPitchedFloatToGrayRGBA_kernel(uchar4 *out_image,
                                                     const float *in_image,
                                                     int width, int height,
                                                     int pitch, float lowerLim,
                                                     float upperLim) {
  const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

  uchar4 temp;

  if (x < width && y < height) {
    //    float val = in_image[__mul24(y,pitch)+x];
    float val = *((float *)((char *)in_image + y * pitch) + x);

    // rescale value from [lowerLim,upperLim] to [0,255]
    val -= lowerLim;
    val /= (upperLim - lowerLim);
    val *= 255.0;

    temp.x = val;
    temp.y = val;
    temp.z = val;
    temp.w = 255;

    out_image[__mul24(y, width) + x] = temp;
  }
}

// Convert float array to RGBA grayscale
__global__ void convertFloatArrayToGrayRGBA_kernel(uchar4 *out_image, int width,
                                                   int height, float lower_lim,
                                                   float upper_lim) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  uchar4 temp;

  if (x < width && y < height) {
    float val = tex2D(d_float_texture0, (float)x + 0.5f, (float)y + 0.5f);

    // rescale value from [lowerLim,upperLim] to [0,255]
    val -= lower_lim;
    val /= (upper_lim - lower_lim);
    val *= 255.0;

    temp.x = val;
    temp.y = val;
    temp.z = val;
    temp.w = 255;

    out_image[y * width + x] = temp;
  }
}

// Merge two grayscale images into RGBA anaglyph

__global__ void createAnaglyph_kernel(uchar4 *out_image,
                                      const float *left_image,
                                      const float *right_image, int width,
                                      int height, int pre_shift) {
  const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  const int x_right = x - pre_shift;
  const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
  uchar4 temp;

  if (x < width && y < height) {

    temp.x = left_image[__mul24(y, width) + x];

    if (x_right > 0 && x_right < width) {
      temp.y = right_image[__mul24(y, width) + x_right];
      temp.z = temp.y;
    } else {
      temp.y = 0;
      temp.z = 0;
    }

    temp.w = 255;

    out_image[__mul24(y, width) + x] = temp;
  }
}

__global__ void createAnaglyph_kernel(uchar4 *out_image,
                                      const uchar4 *left_image,
                                      const uchar4 *right_image, int width,
                                      int height, int pre_shift) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int x_right = x - pre_shift;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  uchar4 temp;

  if (x < width && y < height) {

    temp.x = rgbaToGray(left_image[y * width + x]);

    if (x_right > 0 && x_right < width) {
      temp.y = rgbaToGray(right_image[y * width + x_right]);
      temp.z = temp.y;
    } else {
      temp.y = 0;
      temp.z = 0;
    }

    temp.w = 255;

    out_image[y * width + x] = temp;
  }
}

// convert 2D vectors to an RGBA angle image and an RGBA magnitude image

__global__ void convert2DVectorToAngleMagnitude_kernel(
    uchar4 *d_angle_image, uchar4 *d_magnitude_image, float *d_vector_X,
    float *d_vector_Y, int width, int height, float lower_ang, float upper_ang,
    float lower_mag, float upper_mag) {
  const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
  uchar4 temp_angle, temp_magnitude;

  if (x < width && y < height) {
    float vector_X = d_vector_X[__mul24(y, width) + x];
    float vector_Y = d_vector_Y[__mul24(y, width) + x];

    // compute angle and magnitude
    float angle = atan2f(vector_Y, vector_X);
    float magnitude = vector_X * vector_X + vector_Y * vector_Y;
    magnitude = sqrtf(magnitude);

    // first draw unmatched pixels in white
    if (!isfinite(magnitude)) {
      temp_angle.x = 255;
      temp_angle.y = 255;
      temp_angle.z = 255;
      temp_angle.w = 255;
      temp_magnitude.x = 255;
      temp_magnitude.y = 255;
      temp_magnitude.z = 255;
      temp_magnitude.w = 255;
    } else {
      // rescale angle and magnitude from [lower,upper] to [0,1] and convert to
      // RGBA jet colorspace

      angle -= lower_ang;
      angle /= (upper_ang - lower_ang);

      float r = 1.0f;
      float g = 1.0f;
      float b = 1.0f;

      if (angle < 0.25f) {
        r = 0;
        g = 4.0f * angle;
      } else if (angle < 0.5f) {
        r = 0;
        b = 1.0 + 4.0f * (0.25f - angle);
      } else if (angle < 0.75f) {
        r = 4.0f * (angle - 0.5f);
        b = 0;
      } else {
        g = 1.0f + 4.0f * (0.75f - angle);
        b = 0;
      }

      temp_angle.x = 255.0 * r;
      temp_angle.y = 255.0 * g;
      temp_angle.z = 255.0 * b;
      temp_angle.w = 255;

      magnitude -= lower_mag;
      magnitude /= (upper_mag - lower_mag);

      r = 1.0f;
      g = 1.0f;
      b = 1.0f;

      if (magnitude < 0.25f) {
        r = 0;
        g = 4.0f * magnitude;
      } else if (magnitude < 0.5f) {
        r = 0;
        b = 1.0 + 4.0f * (0.25f - magnitude);
      } else if (magnitude < 0.75f) {
        r = 4.0f * (magnitude - 0.5f);
        b = 0;
      } else {
        g = 1.0f + 4.0f * (0.75f - magnitude);
        b = 0;
      }

      temp_magnitude.x = 255.0 * r;
      temp_magnitude.y = 255.0 * g;
      temp_magnitude.z = 255.0 * b;
      temp_magnitude.w = 255;
    }

    d_angle_image[__mul24(y, width) + x] = temp_angle;
    d_magnitude_image[__mul24(y, width) + x] = temp_magnitude;
  }
}

// threshold floats using lowerLim and upperLim into RGBA black/white image

__global__ void convertFloatToRGBAbinary_kernel(uchar4 *out_image,
                                                const float *in_image,
                                                int width, int height,
                                                float lowerLim,
                                                float upperLim) {
  const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
  uchar4 temp;
  if (x < width && y < height) {
    float val = in_image[__mul24(y, width) + x];

    // draw everything invalid or out of lim in white
    if (!isfinite(val) || (val < lowerLim) || (val > upperLim)) {
      temp.x = 255;
      temp.y = 255;
      temp.z = 255;
      temp.w = 255;
    } else {
      temp.x = 0.0f;
      temp.y = 0.0f;
      temp.z = 0.0f;
      temp.w = 0.0f;
    }
    out_image[__mul24(y, width) + x] = temp;
  }
}

// blend float image with float label (membership specified by lowerLim and
// upperLim)

__global__ void blendFloatImageFloatLabelToRGBA_kernel(
    uchar4 *out_image, const float *in_image, const float *label, int width,
    int height, float lowerLim, float upperLim) {
  const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
  uchar4 temp;
  if (x < width && y < height) {
    unsigned char img =
        (unsigned char)(0.5f * in_image[__mul24(y, width) + x] + 128.0f);
    float val = label[__mul24(y, width) + x];

    // draw everything invalid or out of lim in white
    if (!isfinite(val) || (val < lowerLim) || (val > upperLim)) {
      // don't blend

      temp.x = img;
      temp.y = img;
      temp.z = img;
      temp.w = 255;

    } else {

      // blend

      temp.x = 0.6f * img;
      temp.y = 0.6f * img;
      temp.z = img;
      temp.w = 255;
    }
    out_image[__mul24(y, width) + x] = temp;
  }
}

//__global__ void blendFloatImageFloatArrayToRGBA_kernel(uchar4 *out_image,
// const float *in_image, int width, int height, float w_r, float w_g, float
// w_b)
//{
//  const int x = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
//  const int y = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
//  uchar4 temp;
//  if(x < width && y < height) {
//    unsigned char img = (unsigned char)(in_image[__mul24(y,width)+x]);
//    uchar4 t = tex2D(d_rgba_texture,(float)x + 0.5f,(float)y + 0.5f);
//    float model = rgbaToGray(t);

//    if (t.w==0) { // don't blend
//      temp.x = img;
//      temp.y = img;
//      temp.z = img;

//    } else { // blend

////      model = (model-1.0f)*255.0f;
//      temp.x = w_r*model;
//      temp.y = w_g*img;
//      temp.z = w_b*img;
//    }

//    out_image[__mul24(y,width)+x] = temp;
//  }

//}

__global__ void blendFloatImageRGBAArrayToRGBA_kernel(uchar4 *out_image,
                                                      const float *in_image,
                                                      int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  uchar4 temp;
  if (x < width && y < height) {

    float img = in_image[y * width + x];
    uchar4 t = tex2D(d_rgba_texture, (float)x + 0.5f, (float)y + 0.5f);
    float model = rgbaToGray(t);

    unsigned char delta = (unsigned char)(abs(img - model));
    unsigned char avg = (unsigned char)(0.5f * (img + model));

    // increase image brightness
    unsigned char u_img = (unsigned char)(img);

    if (t.w == 0) { // outside mask -> don't blend

      temp.x = u_img;
      temp.y = u_img;
      temp.z = u_img;

    } else { // blend

      temp.x = 0;
      temp.y = avg;
      temp.z = delta;
    }

    out_image[y * width + x] = temp;
  }
}

__global__ void blendFloatImageFloatArrayToRGBA_kernel(uchar4 *out_image,
                                                       const float *in_image,
                                                       int pitch_out,
                                                       int pitch_in, int width,
                                                       int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  uchar4 temp;
  if (x < width && y < height) {

    float img = *((const float *)((const char *)in_image + y * pitch_in) + x);
    float model = tex2D(d_float_texture0, (float)x + 0.5f, (float)y + 0.5f);
    bool valid = (model != 0.0f);
    model = model - 1.0f;
    model *= 255.0;

    unsigned char delta = (unsigned char)(abs(img - model));
    unsigned char avg = (unsigned char)(0.5f * (img + model));

    // increase image brightness
    unsigned char u_img = (unsigned char)(img);

    if (!valid) { // outside mask (or just black) -> don't blend

      temp.x = u_img;
      temp.y = u_img;
      temp.z = u_img;

    } else { // blend

      temp.x = 0;
      temp.y = avg;
      temp.z = delta;
    }

    *((uchar4 *)((char *)out_image + y * pitch_out) + x) = temp;
  }
}

__global__ void blendMultiColor_kernel(uchar4 *out_image, const float *in_image,
                                       int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  uchar4 temp;
  if (x < width && y < height) {

    float img = in_image[y * width + x];

    // determine gl coord
    float xt = (float)x + 0.5f;
    float yt = (float)y + 0.5f;
    float model = tex2D(d_float_texture0, xt, yt);
    int segment_ind = (int)rintf(tex2D(d_float_texture1, xt, yt));
    bool valid = (model != 0.0f) && (segment_ind != 0);
    model = model - 1.0f;
    model *= 255.0;

    unsigned char u_img = (unsigned char)(img);

    if (!valid) {
      // outside mask (or just black) -> don't blend
      temp.x = u_img;
      temp.y = u_img;
      temp.z = u_img;
    } else if (segment_ind == 10) { // shelf
      temp.x = (unsigned char)(0.8f * img + 0.2f * model);
      temp.y = 0;
      temp.z = 0;
    } else if (segment_ind >= 20) { // robot
      temp.x = 0;
      temp.y = 0;
      temp.z = (unsigned char)(0.8f * img + 0.2f * model);
    } else { // object
      temp.x = 0;
      temp.y = (unsigned char)(0.5f * img + 0.5f * model);
      temp.z = (unsigned char)(abs(img - model));
    }

    out_image[y * width + x] = temp;
  }
}

__global__ void augmentedReality_kernel(float *out_image, const float *in_image,
                                        int width, int height, int pitch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {

    float img_back = *((float *)((char *)in_image + y * pitch) + x);
    uchar4 img_front = tex2D(d_rgba_texture, (float)x + 0.5f, (float)y + 0.5f);
    *((float *)((char *)out_image + y *pitch) + x) =
        (img_front.w == 255) ? rgbaToGray(img_front) : img_back;
  }
}

__global__ void augmentedRealityFloatArray_kernel(float *out_image,
                                                  const float *in_image,
                                                  int width, int height,
                                                  int pitch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {

    float img_back = *((float *)((char *)in_image + y * pitch) + x);
    float img_front = tex2D(d_float_texture0, (float)x + 0.5f, (float)y + 0.5f);
    bool valid = (img_front != 0.0f);
    img_front = img_front - 1.0f;
    img_front *= 255.0;
    *((float *)((char *)out_image + y *pitch) + x) =
        valid ? img_front : img_back;
  }
}

__global__ void augmentedRealityFloatArraySelectiveBlend_kernel(
    float *out_image, const float *in_image, int width, int height, int pitch,
    int max_segment_ind) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {

    float img_back = *((float *)((char *)in_image + y * pitch) + x);

    // determine gl coord
    float xt = (float)x + 0.5f;
    float yt = (float)y + 0.5f;

    float img_front = tex2D(d_float_texture0, xt, yt);
    int segment_ind = (int)rintf(tex2D(d_float_texture1, xt, yt));

    bool valid = (segment_ind > 0) && (segment_ind <= max_segment_ind);
    img_front = img_front - 1.0f;
    img_front *= 255.0;
    *((float *)((char *)out_image + y *pitch) + x) =
        valid ? img_front : img_back;
  }
}

__global__ void colorBlend_kernel(uchar4 *out_image, const uchar4 *in_image,
                                  int width, int height, float alpha_scale) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {

    uchar4 out;
    int IND = y * width + x;

    uchar4 img_back = in_image[IND];
    uchar4 img_front = tex2D(d_rgba_texture, (float)x + 0.5f, (float)y + 0.5f);

    float alpha_front = alpha_scale * 0.0039215686274510f *
                        (float)img_front.w; // divide by 255 and scale
    float alpha_back = 1.0f - alpha_front;

    out.x = alpha_front * img_front.x + alpha_back * img_back.x;
    out.y = alpha_front * img_front.y + alpha_back * img_back.y;
    out.z = alpha_front * img_front.z + alpha_back * img_back.z;
    out.w = 255;

    out_image[IND] = out;
  }
}

__global__ void invalidateFlow_kernel(float *modFlowX, float *modFlowY,
                                      const float *constFlowX,
                                      const float *constFlowY, int width,
                                      int height, float cons_thres) {
  const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

  if (x < width && y < height) {
    int ind = __mul24(y, width) + x;
    float mFX = modFlowX[ind];
    float mFY = modFlowY[ind];
    float cFX = constFlowX[ind];
    float cFY = constFlowY[ind];

    float err = (mFX - cFX) * (mFX - cFX) + (mFY - cFY) * (mFY - cFY);
    err = sqrtf(err);

    if (err > cons_thres) {
      mFX = nanf("");
      mFY = nanf("");
    }

    modFlowX[ind] = mFX;
    modFlowY[ind] = mFY;
  }
}

__global__ void colorInvalids_kernel(uchar4 *out_image, const float *in_image,
                                     int width, int height) {
  const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

  if (x < width && y < height) {
    int ind = __mul24(y, width) + x;
    uchar4 temp = out_image[ind];
    float value = in_image[ind];

    if (!isfinite(value)) { // color
      temp.x *= 0.5f;
      temp.y *= 0.5f;
    }

    out_image[ind] = temp;
  }
}

__global__ void convertKinectDisparityToRegularDisparity_kernel(
    float *d_regularDisparity, int d_regularDisparityPitch,
    const float *d_KinectDisparity, int d_KinectDisparityPitch, int width,
    int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < width) & (y < height)) { // are we in the image?

    float d_in =
        *((float *)((char *)d_KinectDisparity + y * d_KinectDisparityPitch) +
          x);

    float d_out = (d_in == 0.0f) ? nanf("") : -d_in;

    *((float *)((char *)d_regularDisparity + y *d_regularDisparityPitch) + x) =
        d_out;
  }
}

__global__ void convertKinectDisparityInPlace_kernel(float *d_disparity,
                                                     int pitch, int width,
                                                     int height,
                                                     float depth_scale) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < width) & (y < height)) { // are we in the image?

    float *d_in = (float *)((char *)d_disparity + y * pitch) + x;
    *d_in = (*d_in == 0.0f) ? nanf("") : (-depth_scale / *d_in);
  }
}

__global__ void colorDistDiff_kernel(uchar4 *out_image, const float *disparity,
                                     int disparity_pitch,
                                     const float *disparity_prior, int width,
                                     int height, float f, float b, float ox,
                                     float oy, float dist_thres) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {

    int ind = y * width + x;
    uchar4 temp = out_image[ind];
    float disp = *((float *)((char *)disparity + y * disparity_pitch) + x);
    float disp_model = disparity_prior[ind];

    // 3D reconstruct and measure Euclidian distance
    float xt = __fdividef((x - ox), f);
    float yt = -__fdividef((y - oy), f); // coord. transform

    float Zm = -(f * b) / disp_model;
    float Xm = xt * Zm;
    float Ym = yt * Zm;

    float Zd = -(f * b) / disp;
    float Xd = xt * Zd;
    float Yd = yt * Zd;

    float d_md = sqrtf((Xm - Xd) * (Xm - Xd) + (Ym - Yd) * (Ym - Yd) +
                       (Zm - Zd) * (Zm - Zd));

    bool color = (d_md > dist_thres) | (isfinite(disp) & ~isfinite(disp_model));

    if (color) { // color
      temp.x *= 0.5f;
      temp.y *= 0.5f;
    }

    out_image[ind] = temp;
  }
}

// Calling functions
void convertFloatToRGBA(uchar4 *d_out_image, const float *d_in_image, int width,
                        int height, float lowerLim, float upperLim) {

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  convertFloatToRGBA_kernel << <dimGrid, dimBlock>>>
      (d_out_image, d_in_image, width, height, lowerLim, upperLim);
}

void convertPitchedFloatToRGBA(uchar4 *d_out_image, const float *d_in_image,
                               int width, int height, int pitch, float lowerLim,
                               float upperLim) {

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  convertPitchedFloatToRGBA_kernel << <dimGrid, dimBlock>>>
      (d_out_image, d_in_image, width, height, pitch, lowerLim, upperLim);
}

void convertKinectFloatToRGBA(uchar4 *d_out_image, const float *d_in_image,
                              int width, int height, int pitch, float lowerLim,
                              float upperLim) {

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  convertKinectFloatToRGBA_kernel << <dimGrid, dimBlock>>>
      (d_out_image, d_in_image, width, height, pitch, lowerLim, upperLim);
}

void convertFloatToRGBA(uchar4 *d_out_image, const float *d_in_image, int width,
                        int height) {

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  convertFloatToRGBA_kernel << <dimGrid, dimBlock>>>
      (d_out_image, d_in_image, width, height);
}

void convertFlowToRGBA(uchar4 *d_flowx_out, uchar4 *d_flowy_out,
                       const float *d_flowx_in, const float *d_flowy_in,
                       int width, int height, float lowerLim, float upperLim,
                       float minMag) {

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  convertFlowToRGBA_kernel << <dimGrid, dimBlock>>>
      (d_flowx_out, d_flowy_out, d_flowx_in, d_flowy_in, width, height,
       lowerLim, upperLim, minMag);
}

void convertPitchedFloatToGrayRGBA(uchar4 *d_out_image, const float *d_in_image,
                                   int width, int height, int pitch,
                                   float lowerLim, float upperLim) {

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  convertPitchedFloatToGrayRGBA_kernel << <dimGrid, dimBlock>>>
      (d_out_image, d_in_image, width, height, pitch, lowerLim, upperLim);
}

void convertFloatArrayToGrayRGBA(uchar4 *d_out_image, cudaArray *in_array,
                                 int width, int height, float lower_lim,
                                 float upper_lim) {
  // Bind textures to arrays
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();
  cudaBindTextureToArray(d_float_texture0, in_array, channelFloat);

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  convertFloatArrayToGrayRGBA_kernel << <dimGrid, dimBlock>>>
      (d_out_image, width, height, lower_lim, upper_lim);

  cudaUnbindTexture(d_float_texture0);
}

void createAnaglyph(uchar4 *d_out_image, const float *d_left_image,
                    const float *d_right_image, int width, int height,
                    int pre_shift) {

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  createAnaglyph_kernel << <dimGrid, dimBlock>>>
      (d_out_image, d_left_image, d_right_image, width, height, pre_shift);
}

void createAnaglyph(uchar4 *d_out_image, const uchar4 *d_left_image,
                    const uchar4 *d_right_image, int width, int height,
                    int pre_shift) {

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  createAnaglyph_kernel << <dimGrid, dimBlock>>>
      (d_out_image, d_left_image, d_right_image, width, height, pre_shift);
}

void convert2DVectorToAngleMagnitude(uchar4 *d_angle_image,
                                     uchar4 *d_magnitude_image,
                                     float *d_vector_X, float *d_vector_Y,
                                     int width, int height, float lower_ang,
                                     float upper_ang, float lower_mag,
                                     float upper_mag) {

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  convert2DVectorToAngleMagnitude_kernel << <dimGrid, dimBlock>>>
      (d_angle_image, d_magnitude_image, d_vector_X, d_vector_Y, width, height,
       lower_ang, upper_ang, lower_mag, upper_mag);
}

void convertFloatToRGBAbinary(uchar4 *out_image, const float *in_image,
                              int width, int height, float lowerLim,
                              float upperLim) {

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  convertFloatToRGBAbinary_kernel << <dimGrid, dimBlock>>>
      (out_image, in_image, width, height, lowerLim, upperLim);
}

void mutuallyValidateFlowStereo(float *d_flow, float *d_disparity, int width,
                                int height) {
  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  matchValidity_kernel << <dimGrid, dimBlock>>>
      (d_flow, d_disparity, width, height);
}

void deInterleave(float *d_X_out, float *d_Y_out, float2 *d_XY_in,
                  int pitch_out, int pitch_in, int width, int height) {
  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  //   deInterleave_kernel<<<dimGrid,dimBlock>>>(d_X_out, d_Y_out, d_XY_in,
  // pitch_out, pitch_in, width, height);

  deInterleave_kernel2 << <dimGrid, dimBlock>>>
      (d_X_out, d_Y_out, (char *)d_XY_in, pitch_out, pitch_in, width, height);
}

void applyIMOMask(float *d_IMOMask, float *d_IMO, const float *d_disparity,
                  float offset, int width, int height) {
  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  IMOMask_kernel << <dimGrid, dimBlock>>>
      (d_IMOMask, d_IMO, d_disparity, offset, width, height);
}

void blendFloatImageFloatLabelToRGBA(uchar4 *out_image, const float *in_image,
                                     const float *label, int width, int height,
                                     float lowerLim, float upperLim) {

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  blendFloatImageFloatLabelToRGBA_kernel << <dimGrid, dimBlock>>>
      (out_image, in_image, label, width, height, lowerLim, upperLim);
}

void blendFloatImageRGBAArrayToRGBA(uchar4 *out_image, const float *in_image,
                                    cudaArray *in_array, int width, int height,
                                    float w_r, float w_g, float w_b) {

  // Bind textures to arrays
  cudaChannelFormatDesc channelUChar4 = cudaCreateChannelDesc<uchar4>();
  cudaBindTextureToArray(d_rgba_texture, in_array, channelUChar4);

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  blendFloatImageRGBAArrayToRGBA_kernel << <dimGrid, dimBlock>>>
      (out_image, in_image, width, height);

  cudaUnbindTexture(d_rgba_texture);
}

void blendFloatImageFloatArrayToRGBA(uchar4 *out_image, const float *in_image,
                                     cudaArray *in_array, int pitch_out,
                                     int pitch_in, int width, int height) {
  // Bind textures to arrays
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();
  cudaBindTextureToArray(d_float_texture0, in_array, channelFloat);

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  blendFloatImageFloatArrayToRGBA_kernel << <dimGrid, dimBlock>>>
      (out_image, in_image, pitch_out, pitch_in, width, height);

  cudaUnbindTexture(d_float_texture0);
}

void blendMultiColor(uchar4 *out_image, const float *in_image,
                     cudaArray *in_texture, cudaArray *in_segment_index,
                     int width, int height) {

  // Bind textures to arrays
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();
  cudaBindTextureToArray(d_float_texture0, in_texture, channelFloat);
  cudaBindTextureToArray(d_float_texture1, in_segment_index, channelFloat);

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  blendMultiColor_kernel << <dimGrid, dimBlock>>>
      (out_image, in_image, width, height);

  cudaUnbindTexture(d_float_texture1);
  cudaUnbindTexture(d_float_texture0);
}

void augmentedReality(float *out_image, const float *in_image,
                      cudaArray *in_array, int width, int height, int pitch) {

  // Bind textures to arrays
  cudaChannelFormatDesc channelUChar4 = cudaCreateChannelDesc<uchar4>();
  cudaBindTextureToArray(d_rgba_texture, in_array, channelUChar4);

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  augmentedReality_kernel << <dimGrid, dimBlock>>>
      (out_image, in_image, width, height, pitch);

  cudaUnbindTexture(d_rgba_texture);
}

void augmentedRealityFloatArray(float *out_image, const float *in_image,
                                cudaArray *in_array, int width, int height,
                                int pitch) {

  // Bind textures to arrays
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();
  cudaBindTextureToArray(d_float_texture0, in_array, channelFloat);

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  augmentedRealityFloatArray_kernel << <dimGrid, dimBlock>>>
      (out_image, in_image, width, height, pitch);

  cudaUnbindTexture(d_float_texture0);
}

void augmentedRealityFloatArraySelectiveBlend(float *out_image,
                                              const float *in_image,
                                              const cudaArray *texture,
                                              const cudaArray *segment_index,
                                              int width, int height, int pitch,
                                              int max_segment_ind) {
  // Bind textures to arrays
  cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();
  cudaBindTextureToArray(d_float_texture0, texture, channelFloat);
  cudaBindTextureToArray(d_float_texture1, segment_index, channelFloat);

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  augmentedRealityFloatArraySelectiveBlend_kernel << <dimGrid, dimBlock>>>
      (out_image, in_image, width, height, pitch, max_segment_ind);

  cudaUnbindTexture(d_float_texture1);
  cudaUnbindTexture(d_float_texture0);
}

void colorBlend(uchar4 *out_image, const uchar4 *in_image, cudaArray *in_array,
                int width, int height, float alpha_scale) {

  // Bind texture to array
  cudaChannelFormatDesc channelUChar4 = cudaCreateChannelDesc<uchar4>();
  cudaBindTextureToArray(d_rgba_texture, in_array, channelUChar4);

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  colorBlend_kernel << <dimGrid, dimBlock>>>
      (out_image, in_image, width, height, alpha_scale);

  cudaUnbindTexture(d_rgba_texture);
}

void invalidateFlow(float *modFlowX, float *modFlowY, const float *constFlowX,
                    const float *constFlowY, int width, int height,
                    float cons_thres) {

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  invalidateFlow_kernel << <dimGrid, dimBlock>>>
      (modFlowX, modFlowY, constFlowX, constFlowY, width, height, cons_thres);
}

void colorInvalids(uchar4 *out_image, const float *in_image, int width,
                   int height) {
  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  colorInvalids_kernel << <dimGrid, dimBlock>>>
      (out_image, in_image, width, height);
}

void convertKinectDisparityToRegularDisparity(float *d_regularDisparity,
                                              int d_regularDisparityPitch,
                                              const float *d_KinectDisparity,
                                              int d_KinectDisparityPitch,
                                              int width, int height) {

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  convertKinectDisparityToRegularDisparity_kernel << <dimGrid, dimBlock>>>
      (d_regularDisparity, d_regularDisparityPitch, d_KinectDisparity,
       d_KinectDisparityPitch, width, height);
}

void convertKinectDisparityInPlace(float *d_disparity, int pitch, int width,
                                   int height, float depth_scale) {
  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  convertKinectDisparityInPlace_kernel << <dimGrid, dimBlock>>>
      (d_disparity, pitch, width, height, depth_scale);
}

void colorDistDiff(uchar4 *out_image, const float *disparity,
                   int disparity_pitch, const float *disparity_prior, int width,
                   int height, float focal_length, float baseline,
                   float nodal_point_x, float nodal_point_y, float dist_thres) {
  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  colorDistDiff_kernel << <dimGrid, dimBlock>>>
      (out_image, disparity, disparity_pitch, disparity_prior, width, height,
       focal_length, baseline, nodal_point_x, nodal_point_y, dist_thres);
}

} // end namespace vision
