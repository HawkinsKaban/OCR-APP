package org.tensorflow.lite.examples.ocr

import android.app.Application
import android.util.Log

class OCRApplication : Application() {
  override fun onCreate() {
    super.onCreate()

    // Initialize OpenCV at application start
    try {
      // Use the OpenCVHelper to initialize
      val success = OpenCVHelper.initOpenCV(this)
      if (success) {
        Log.d(TAG, "OpenCV initialized successfully")
      } else {
        Log.e(TAG, "Failed to initialize OpenCV")
      }
    } catch (e: Exception) {
      Log.e(TAG, "Error initializing OpenCV: ${e.message}")
      e.printStackTrace()
    }
  }

  companion object {
    private const val TAG = "OCRApplication"
  }
}