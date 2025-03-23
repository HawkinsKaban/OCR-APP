package org.tensorflow.lite.examples.ocr

import android.app.Application
import android.util.Log

class OCRApplication : Application() {
  override fun onCreate() {
    super.onCreate()

    // Initialize OpenCV at application start
    try {
      // Load OpenCV library
      System.loadLibrary("opencv_java4")
      Log.d(TAG, "OpenCV library loaded successfully")
    } catch (e: Exception) {
      Log.e(TAG, "Error loading OpenCV: ${e.message}")
      e.printStackTrace()
    }
  }

  companion object {
    private const val TAG = "OCRApplication"
  }
}