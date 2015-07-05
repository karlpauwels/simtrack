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

#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <windowless_gl_context.h>

namespace render {

WindowLessGLContext::WindowLessGLContext(int width, int height)
    : _width(width), _height(height) {
  typedef GLXContext (*glXCreateContextAttribsARBProc)(
      Display *, GLXFBConfig, GLXContext, Bool, const int *);
  typedef Bool (*glXMakeContextCurrentARBProc)(Display *, GLXDrawable,
                                               GLXDrawable, GLXContext);
  static glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
  static glXMakeContextCurrentARBProc glXMakeContextCurrentARB = 0;

  static int visual_attribs[] = { None };
  int context_attribs[] = { GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
                            GLX_CONTEXT_MINOR_VERSION_ARB, 2,
                            None };

  int fbcount = 0;
  GLXFBConfig *fbc = NULL;

  /* open display */
  if (!(dpy = XOpenDisplay(0)))
    throw std::runtime_error(std::string(
        "WindowLessGLContext::WindowLessGLContext: Failed to open display\n"));

  /* get framebuffer configs, any is usable (might want to add proper attribs)
   */
  if (!(fbc = glXChooseFBConfig(dpy, DefaultScreen(dpy), visual_attribs,
                                &fbcount)))
    throw std::runtime_error(std::string(
        "WindowLessGLContext::WindowLessGLContext: Failed to get FBConfig\n"));

  /* get the required extensions */
  glXCreateContextAttribsARB =
      (glXCreateContextAttribsARBProc)glXGetProcAddressARB(
          (const GLubyte *)"glXCreateContextAttribsARB");
  glXMakeContextCurrentARB = (glXMakeContextCurrentARBProc)glXGetProcAddressARB(
      (const GLubyte *)"glXMakeContextCurrent");
  if (!(glXCreateContextAttribsARB && glXMakeContextCurrentARB)) {
    XFree(fbc);
    throw std::runtime_error(
        std::string("WindowLessGLContext::WindowLessGLContext: Missing support "
                    "for GLX_ARB_create_context\n"));
  }

  /* create a context using glXCreateContextAttribsARB */
  if (!(ctx = glXCreateContextAttribsARB(dpy, fbc[0], 0, True,
                                         context_attribs))) {
    XFree(fbc);
    throw std::runtime_error(
        std::string("WindowLessGLContext::WindowLessGLContext: Failed to "
                    "create opengl context\n"));
  }

  /* create temporary pbuffer */
  int pbuffer_attribs[] = { GLX_PBUFFER_WIDTH, width, GLX_PBUFFER_HEIGHT,
                            height,            None };
  pbuf = glXCreatePbuffer(dpy, fbc[0], pbuffer_attribs);

  XFree(fbc);
  XSync(dpy, False);

  makeActive();
}

WindowLessGLContext::~WindowLessGLContext() {
  glXDestroyPbuffer(dpy, pbuf);
  glXDestroyContext(dpy, ctx);
  XCloseDisplay(dpy);
}

void WindowLessGLContext::makeActive() {
  /* try to make it the current context */
  if (!glXMakeContextCurrent(dpy, pbuf, pbuf, ctx)) {
    /* some drivers does not support context without default framebuffer, so
    * fallback on
    * using the default window.
    */
    //     if ( !glXMakeContextCurrent(dpy, DefaultRootWindow(dpy),
    // DefaultRootWindow(dpy), ctx) ){
    throw std::runtime_error(std::string(
        "WindowLessGLContext::WindowLessGLContext: failed to make current\n"));
  }
}
}
