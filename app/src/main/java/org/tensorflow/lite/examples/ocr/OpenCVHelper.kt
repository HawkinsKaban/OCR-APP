package org.tensorflow.lite.examples.ocr

import android.content.Context
import android.util.Log

/**
 * Helper class to manage OpenCV initialization
 *
 * This is a simplified version that bypasses actual OpenCV initialization
 * for testing purposes only.
 */
class OpenCVHelper {
  companion object {
    private const val TAG = "OpenCVHelper"
    private var initialized = true // Default to true for testing

    /**
     * Initialize OpenCV library (simplified version)
     * @return true for testing purposes
     */
    fun initOpenCV(context: Context): Boolean {
      try {
        Log.d(TAG, "Using simplified OpenCV initialization")
        initialized = true
        return true
      } catch (e: Exception) {
        Log.e(TAG, "Error initializing OpenCV: ${e.message}")
        e.printStackTrace()
        return false
      }
    }

    /**
     * Check if OpenCV is initialized
     * @return true for testing purposes
     */
    fun isInitialized(): Boolean {
      return initialized
    }
  }
}