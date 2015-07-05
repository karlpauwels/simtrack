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

#include <OgreConfigFile.h>
#include <OgreWindowEventUtilities.h>
#include <ros/package.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/glx.h>

#include <ogre_context.h>
#include <env_config.h>

namespace render {

OgreContext::OgreContext() {
  setupDummyWindowId();

  Ogre::LogManager *log_manager = Ogre::LogManager::getSingletonPtr();
  if (log_manager == NULL)
    log_manager = new Ogre::LogManager();
  log_manager->createLog("Ogre.log", false, false, false);

  std::string package_path = ros::package::getPath(ROS_PACKAGE_NAME);
  ogre_root_ = std::unique_ptr<Ogre::Root>{ new Ogre::Root(
      package_path + "/ogre_media/plugins.cfg") };

  std::string plugin_prefix = render::get_ogre_plugin_path() + "/";
#ifdef Q_OS_MAC
  plugin_prefix += "lib";
#endif
  ogre_root_->loadPlugin(plugin_prefix + "RenderSystem_GL");

  Ogre::RenderSystem *rs =
      ogre_root_->getRenderSystemByName("OpenGL Rendering Subsystem");
  ogre_root_->setRenderSystem(rs);

  Ogre::NameValuePairList opts;
  opts["hidden"] = "true";
  ogre_root_->initialise(false);
  makeRenderWindow(dummy_window_id_, 1, 1);

  // Set default mipmap level (note: some APIs ignore this)
  Ogre::TextureManager::getSingleton().setDefaultNumMipmaps(5);

  // setup resources
  Ogre::ResourceGroupManager::getSingleton().addResourceLocation(
      package_path + "/ogre_media/models", "FileSystem", ROS_PACKAGE_NAME);
  Ogre::ResourceGroupManager::getSingleton().addResourceLocation(
      package_path + "/ogre_media/materials/scripts", "FileSystem",
      ROS_PACKAGE_NAME);
  Ogre::ResourceGroupManager::getSingleton().addResourceLocation(
      package_path + "/ogre_media/materials/glsl", "FileSystem",
      ROS_PACKAGE_NAME);

  Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();

  scene_manager_ = ogre_root_->createSceneManager("DefaultSceneManager");
}

OgreContext::~OgreContext() {}

void OgreContext::setupDummyWindowId() {
  Display *display = XOpenDisplay(0);
  assert(display);

  int screen = DefaultScreen(display);

  int attribList[] = { GLX_RGBA,         GLX_DOUBLEBUFFER, GLX_DEPTH_SIZE, 16,
                       GLX_STENCIL_SIZE, 8,                None };

  XVisualInfo *visual = glXChooseVisual(display, screen, (int *)attribList);

  dummy_window_id_ = XCreateSimpleWindow(display, RootWindow(display, screen),
                                         0, 0, 1, 1, 0, 0, 0);

  GLXContext context = glXCreateContext(display, visual, NULL, 1);

  glXMakeCurrent(display, dummy_window_id_, context);
}

// On Intel graphics chips under X11, there sometimes comes a
// BadDrawable error during Ogre render window creation.  It is not
// consistent, happens sometimes but not always.  Underlying problem
// seems to be a driver bug.  My workaround here is to notice when
// that specific BadDrawable error happens on request 136 minor 3
// (which is what the problem looks like when it happens) and just try
// over and over again until it works (or until 100 failures, which
// makes it seem like it is a different bug).
static bool x_baddrawable_error = false;
#ifdef Q_WS_X11
static int (*old_error_handler)(Display *, XErrorEvent *);
int checkBadDrawable(Display *display, XErrorEvent *error) {
  if (error->error_code == BadDrawable && error->request_code == 136 &&
      error->minor_code == 3) {
    x_baddrawable_error = true;
    return 0;
  } else {
    // If the error does not exactly match the one from the driver bug,
    // handle it the normal way so we see it.
    return old_error_handler(display, error);
  }
}
#endif // Q_WS_X11

Ogre::RenderWindow *OgreContext::makeRenderWindow(intptr_t window_id,
                                                  unsigned int width,
                                                  unsigned int height) {
  static int windowCounter = 0; // Every RenderWindow needs a unique name, oy.

  Ogre::NameValuePairList params;
  Ogre::RenderWindow *window = NULL;

  std::stringstream window_handle_stream;
  window_handle_stream << window_id;

#ifdef Q_OS_MAC
  params["externalWindowHandle"] = window_handle_stream.str();
#else
  params["parentWindowHandle"] = window_handle_stream.str();
#endif

  params["externalGLControl"] = true;

// Set the macAPI for Ogre based on the Qt implementation
#ifdef QT_MAC_USE_COCOA
  params["macAPI"] = "cocoa";
  params["macAPICocoaUseNSView"] = "true";
#else
  params["macAPI"] = "carbon";
#endif

  std::ostringstream stream;
  stream << "OgreWindow(" << windowCounter++ << ")";

// don't bother trying stereo if Ogre does not support it.
#if !OGRE_STEREO_ENABLE
  bool force_no_stereo_ = true;
#endif

  // attempt to create a stereo window
  bool is_stereo = false;
  if (!force_no_stereo_) {
    params["stereoMode"] = "Frame Sequential";
    window = tryMakeRenderWindow(stream.str(), width, height, &params, 100);
    params.erase("stereoMode");

    if (window) {
#if OGRE_STEREO_ENABLE
      is_stereo = window->isStereoEnabled();
#endif
      if (!is_stereo) {
        // Created a non-stereo window.  Discard it and try again (below)
        // without the stereo parameter.
        ogre_root_->detachRenderTarget(window);
        window->destroy();
        window = NULL;
        stream << "x";
        is_stereo = false;
      }
    }
  }

  if (window == NULL) {
    window = tryMakeRenderWindow(stream.str(), width, height, &params, 100);
  }

  if (window == NULL) {
    //    ROS_ERROR( "Unable to create the rendering window after 100 tries." );
    assert(false);
  }

  if (window) {
    window->setActive(true);
    // window->setVisible(true);
    window->setAutoUpdated(false);
  }

  bool stereo_supported_ = is_stereo;

  //  ROS_INFO_ONCE("Stereo is %s", stereo_supported_ ? "SUPPORTED" : "NOT
  // SUPPORTED");

  return window;
}

Ogre::RenderWindow *OgreContext::tryMakeRenderWindow(
    const std::string &name, unsigned int width, unsigned int height,
    const Ogre::NameValuePairList *params, int max_attempts) {
  Ogre::RenderWindow *window = NULL;
  int attempts = 0;

#ifdef Q_WS_X11
  old_error_handler = XSetErrorHandler(&checkBadDrawable);
#endif

  while (window == NULL && (attempts++) < max_attempts) {
    try {
      window =
          ogre_root_->createRenderWindow(name, width, height, false, params);

      // If the driver bug happened, tell Ogre we are done with that
      // window and then try again.
      if (x_baddrawable_error) {
        ogre_root_->detachRenderTarget(window);
        window = NULL;
        x_baddrawable_error = false;
      }
    }
    catch (std::exception ex) {
      std::cerr << "rviz::RenderSystem: error creating render window: "
                << ex.what() << std::endl;
      window = NULL;
    }
  }

#ifdef Q_WS_X11
  XSetErrorHandler(old_error_handler);
#endif

  if (window && attempts > 1) {
    //    ROS_INFO( "Created render window after %d attempts.", attempts );
  }

  return window;
}
}
