package org.tensorflow.lite.examples.ocr

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.ViewModelProvider
import com.bumptech.glide.Glide
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import kotlinx.coroutines.*
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executors

private const val TAG = "MainActivity"
private const val REQUEST_PERMISSIONS = 1

class MainActivity : AppCompatActivity() {

  private lateinit var viewModel: MLExecutionViewModel
  private lateinit var photoFile: File
  private lateinit var photoUri: Uri

  private var previewBitmap: Bitmap? = null
  private var useGPU = false
  private var ocrModel: OCRModelExecutor? = null
  private val inferenceThread = Executors.newSingleThreadExecutor().asCoroutineDispatcher()
  private val mainScope = MainScope()
  private val mutex = Mutex()

  // OpenCV loader callback
  private val mLoaderCallback = object : BaseLoaderCallback(this) {
    override fun onManagerConnected(status: Int) {
      when (status) {
        LoaderCallbackInterface.SUCCESS -> {
          Log.i(TAG, "OpenCV loaded successfully")
          // Initialize OCR model after OpenCV is loaded
          mainScope.launch(inferenceThread) {
            createModelExecutor(useGPU)
          }
        }
        else -> {
          super.onManagerConnected(status)
          Log.e(TAG, "OpenCV initialization failed: $status")
          Toast.makeText(
            this@MainActivity,
            "Failed to initialize OpenCV. Status: $status",
            Toast.LENGTH_LONG
          ).show()
        }
      }
    }
  }

  // View binding
  private val cameraButton by lazy { findViewById<com.google.android.material.button.MaterialButton>(R.id.camera_button) }
  private val galleryButton by lazy { findViewById<com.google.android.material.button.MaterialButton>(R.id.gallery_button) }
  private val fileButton by lazy { findViewById<com.google.android.material.button.MaterialButton>(R.id.file_button) }
  private val previewImage by lazy { findViewById<android.widget.ImageView>(R.id.preview_image) }
  private val useGpuSwitch by lazy { findViewById<android.widget.Switch>(R.id.switch_use_gpu) }
  private val runOcrButton by lazy { findViewById<com.google.android.material.button.MaterialButton>(R.id.run_ocr_button) }
  private val resultText by lazy { findViewById<android.widget.TextView>(R.id.result_text) }
  private val copyButton by lazy { findViewById<com.google.android.material.button.MaterialButton>(R.id.copy_button) }
  private val shareButton by lazy { findViewById<com.google.android.material.button.MaterialButton>(R.id.share_button) }
  private val toolbar by lazy { findViewById<androidx.appcompat.widget.Toolbar>(R.id.toolbar) }

  private val requiredPermissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
    arrayOf(
      Manifest.permission.CAMERA,
      Manifest.permission.READ_MEDIA_IMAGES
    )
  } else {
    arrayOf(
      Manifest.permission.CAMERA,
      Manifest.permission.READ_EXTERNAL_STORAGE
    )
  }

  private val takePictureLauncher = registerForActivityResult(
    ActivityResultContracts.TakePicture()
  ) { success ->
    if (success) {
      loadBitmapFromUri(photoUri)
    }
  }

  private val pickImageLauncher = registerForActivityResult(
    ActivityResultContracts.GetContent()
  ) { uri ->
    uri?.let { loadBitmapFromUri(it) }
  }

  private val pickFileLauncher = registerForActivityResult(
    ActivityResultContracts.GetContent()
  ) { uri ->
    uri?.let { processDocumentFile(it) }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    setSupportActionBar(toolbar)
    supportActionBar?.title = "OCR App"

    // Initialize PDFBox for Android
    PDFBoxResourceLoader.init(applicationContext)

    viewModel = ViewModelProvider(this)[MLExecutionViewModel::class.java]

    // Observe OCR results
    viewModel.resultingBitmap.observe(this) { resultImage ->
      if (resultImage != null) {
        processOcrResult(resultImage)
      } else {
        runOcrButton.isEnabled = true
        Toast.makeText(this, "Failed to process image", Toast.LENGTH_SHORT).show()
      }
    }

    // Check for permissions
    if (!hasRequiredPermissions()) {
      requestPermissions()
    }

    // Initialize UI and listeners
    setupUI()

    // List assets to verify model files
    try {
      val assetFiles = assets.list("")
      Log.d(TAG, "Assets folder contains: ${assetFiles?.joinToString(", ")}")

      checkModelFile("text_detection.tflite")
      checkModelFile("text_recognition.tflite")
    } catch (e: Exception) {
      Log.e(TAG, "Error listing assets: ${e.message}")
    }

    // Model initialization will be done in onResume after OpenCV is loaded
  }

  private fun checkModelFile(fileName: String) {
    try {
      val assetFileDescriptor = assets.openFd(fileName)
      val fileSize = assetFileDescriptor.length
      assetFileDescriptor.close()
      Log.d(TAG, "$fileName size: ${fileSize / 1024} KB")
    } catch (e: Exception) {
      Log.e(TAG, "Failed to access $fileName: ${e.message}")
    }
  }

  override fun onResume() {
    super.onResume()

    try {
      Log.d(TAG, "Attempting to initialize OpenCV in MainActivity.onResume()...")

      // Load OpenCV
      System.loadLibrary("opencv_java4")

      // Force success callback since we're using the included library
      mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)

    } catch (e: Exception) {
      Log.e(TAG, "Exception during OpenCV initialization in MainActivity: ${e.message}")
      e.printStackTrace()

      // Show error dialog with retry option
      MainScope().launch {
        MaterialAlertDialogBuilder(this@MainActivity)
          .setTitle("OpenCV Initialization Error")
          .setMessage("Error: ${e.message}\nTrying to load native OpenCV library manually failed.")
          .setPositiveButton("Retry") { _, _ ->
            recreate() // Restart the activity to retry OpenCV initialization
          }
          .setNegativeButton("Exit") { _, _ ->
            finish() // Close app
          }
          .show()
      }
    }
  }

  private fun setupUI() {
    cameraButton.setOnClickListener {
      if (hasRequiredPermissions()) {
        dispatchTakePictureIntent()
      } else {
        requestPermissions()
      }
    }

    galleryButton.setOnClickListener {
      if (hasRequiredPermissions()) {
        pickImageLauncher.launch("image/*")
      } else {
        requestPermissions()
      }
    }

    fileButton.setOnClickListener {
      if (hasRequiredPermissions()) {
        pickFileLauncher.launch("application/pdf")
      } else {
        requestPermissions()
      }
    }

    useGpuSwitch.setOnCheckedChangeListener { _, isChecked ->
      useGPU = isChecked
      mainScope.launch(inferenceThread) {
        createModelExecutor(useGPU)
      }
    }

    runOcrButton.setOnClickListener {
      if (previewBitmap != null) {
        runOCR()
      } else {
        Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show()
      }
    }

    copyButton.setOnClickListener {
      val text = resultText.text.toString()
      if (text.isNotBlank() && text != getString(R.string.tfe_ocr_no_text_found)) {
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText("OCR Result", text)
        clipboard.setPrimaryClip(clip)
        Toast.makeText(this, "Text copied to clipboard", Toast.LENGTH_SHORT).show()
      }
    }

    shareButton.setOnClickListener {
      val text = resultText.text.toString()
      if (text.isNotBlank() && text != getString(R.string.tfe_ocr_no_text_found)) {
        val shareIntent = Intent(Intent.ACTION_SEND)
        shareIntent.type = "text/plain"
        shareIntent.putExtra(Intent.EXTRA_TEXT, text)
        startActivity(Intent.createChooser(shareIntent, "Share OCR Result"))
      }
    }
  }

  private suspend fun createModelExecutor(useGPU: Boolean) {
    mutex.withLock {
      ocrModel?.close()
      ocrModel = null

      try {
        Log.d(TAG, "Starting OCR model initialization with GPU: $useGPU")

        // Check if model files exist in assets
        try {
          val assetFiles = assets.list("")
          Log.d(TAG, "Assets folder contains: ${assetFiles?.joinToString(", ")}")

          val hasDetectionModel = assetFiles?.contains("text_detection.tflite") ?: false
          val hasRecognitionModel = assetFiles?.contains("text_recognition.tflite") ?: false

          if (!hasDetectionModel || !hasRecognitionModel) {
            throw IOException("Missing required model files. Detection: $hasDetectionModel, Recognition: $hasRecognitionModel")
          }
        } catch (e: Exception) {
          Log.e(TAG, "Failed to check model files: ${e.message}")
          throw e
        }

        // Create model with more detailed logging
        withContext(Dispatchers.IO) {
          try {
            ocrModel = OCRModelExecutor(this@MainActivity, useGPU)
            Log.d(TAG, "OCR model initialization successful")
          } catch (e: Exception) {
            Log.e(TAG, "Error during OCRModelExecutor initialization: ${e.message}")
            e.printStackTrace()
            throw e
          }
        }

        mainScope.launch(Dispatchers.Main) {
          Toast.makeText(
            this@MainActivity,
            "Model initialized with GPU: $useGPU",
            Toast.LENGTH_SHORT
          ).show()
        }
      } catch (e: Exception) {
        Log.e(TAG, "Failed to create OCRModelExecutor: ${e.message}")
        e.printStackTrace()

        mainScope.launch(Dispatchers.Main) {
          Toast.makeText(
            this@MainActivity,
            "Failed to initialize model: ${e.message}",
            Toast.LENGTH_LONG
          ).show()

          // Show dialog with option to retry initialization
          MaterialAlertDialogBuilder(this@MainActivity)
            .setTitle("Model Initialization Failed")
            .setMessage("Error: ${e.message}\n\nWould you like to retry?")
            .setPositiveButton("Retry") { _, _ ->
              mainScope.launch(inferenceThread) {
                createModelExecutor(useGPU)
              }
            }
            .setNegativeButton("Cancel", null)
            .show()
        }
      }
    }
  }

  private fun runOCR() {
    previewBitmap?.let { bitmap ->
      runOcrButton.isEnabled = false
      resultText.text = "Processing..."

      mainScope.launch(inferenceThread) {
        mutex.withLock {
          if (ocrModel != null) {
            try {
              val result = ocrModel?.execute(bitmap)
              if (result == null) {
                Log.e(TAG, "Model execution returned null result")
                mainScope.launch(Dispatchers.Main) {
                  resultText.text = "Model execution returned null result"
                  runOcrButton.isEnabled = true
                }
                return@withLock
              }

              mainScope.launch(Dispatchers.Main) {
                processOcrResult(result)
              }
            } catch (e: Exception) {
              Log.e(TAG, "Error during OCR execution: ${e.message}")
              e.printStackTrace()
              mainScope.launch(Dispatchers.Main) {
                resultText.text = "Error: ${e.message}"
                runOcrButton.isEnabled = true
              }
            }
          } else {
            Log.e(TAG, "OCR model is null, initialization failed")
            mainScope.launch(Dispatchers.Main) {
              resultText.text = "OCR model not initialized"
              runOcrButton.isEnabled = true

              // Show dialog with option to retry initialization
              MaterialAlertDialogBuilder(this@MainActivity)
                .setTitle("Model Not Initialized")
                .setMessage("The OCR model failed to initialize. Would you like to try again?")
                .setPositiveButton("Retry") { _, _ ->
                  mainScope.launch(inferenceThread) {
                    createModelExecutor(useGPU)
                  }
                }
                .setNegativeButton("Cancel", null)
                .show()
            }
          }
        }
      }
    }
  }

  private fun processOcrResult(result: ModelExecutionResult) {
    runOcrButton.isEnabled = true

    // Display the resulting image with detected text
    Glide.with(this)
      .load(result.bitmapResult)
      .into(previewImage)

    // Process the OCR results
    if (result.itemsFound.isEmpty()) {
      resultText.text = getString(R.string.tfe_ocr_no_text_found)
    } else {
      val allText = result.itemsFound.keys.joinToString("\n")
      resultText.text = allText
    }
  }

  private fun loadBitmapFromUri(uri: Uri) {
    try {
      val inputStream = contentResolver.openInputStream(uri)
      previewBitmap = BitmapFactory.decodeStream(inputStream)

      // Display the selected image
      Glide.with(this)
        .load(uri)
        .into(previewImage)

      // Reset result text
      resultText.text = "Select 'Run OCR' to process the image"

    } catch (e: Exception) {
      Log.e(TAG, "Error loading bitmap: ${e.message}")
      Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
    }
  }

  private fun processDocumentFile(uri: Uri) {
    try {
      val fileExtension = contentResolver.getType(uri)

      if (fileExtension?.contains("pdf") == true) {
        // Process PDF file
        extractTextFromPDF(uri)
      } else {
        Toast.makeText(this, "Unsupported file format", Toast.LENGTH_SHORT).show()
      }
    } catch (e: Exception) {
      Log.e(TAG, "Error processing document: ${e.message}")
      Toast.makeText(this, "Failed to process document: ${e.message}", Toast.LENGTH_SHORT).show()
    }
  }

  private fun extractTextFromPDF(uri: Uri) {
    // Show loading indicator
    resultText.text = "Extracting text from PDF..."

    // Process PDF file in background
    CoroutineScope(Dispatchers.IO).launch {
      try {
        val inputStream = contentResolver.openInputStream(uri)
        val document = PDDocument.load(inputStream)
        val pdfStripper = PDFTextStripper()
        val text = pdfStripper.getText(document)
        document.close()

        // Update UI on main thread
        withContext(Dispatchers.Main) {
          if (text.isBlank()) {
            resultText.text = "No text found in PDF"
          } else {
            resultText.text = text
          }

          // Create a placeholder image for PDF
          val pdfPlaceholder = Bitmap.createBitmap(800, 600, Bitmap.Config.ARGB_8888)
          val canvas = android.graphics.Canvas(pdfPlaceholder)
          canvas.drawColor(android.graphics.Color.LTGRAY)
          val paint = android.graphics.Paint()
          paint.color = android.graphics.Color.BLACK
          paint.textSize = 50f
          canvas.drawText("PDF Document", 50f, 300f, paint)

          previewBitmap = pdfPlaceholder
          previewImage.setImageBitmap(pdfPlaceholder)
        }
      } catch (e: Exception) {
        Log.e(TAG, "Error extracting text from PDF: ${e.message}")
        withContext(Dispatchers.Main) {
          resultText.text = "Failed to extract text from PDF: ${e.message}"
        }
      }
    }
  }

  private fun dispatchTakePictureIntent() {
    try {
      val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
      val storageDir = getExternalFilesDir(null)
      photoFile = File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir)

      photoUri = FileProvider.getUriForFile(
        this,
        "${applicationContext.packageName}.fileprovider",
        photoFile
      )

      takePictureLauncher.launch(photoUri)
    } catch (e: IOException) {
      Log.e(TAG, "Error creating file: ${e.message}")
      Toast.makeText(this, "Error creating file", Toast.LENGTH_SHORT).show()
    }
  }

  private fun hasRequiredPermissions(): Boolean {
    return requiredPermissions.all {
      ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }
  }

  private fun requestPermissions() {
    ActivityCompat.requestPermissions(this, requiredPermissions, REQUEST_PERMISSIONS)
  }

  override fun onRequestPermissionsResult(
    requestCode: Int,
    permissions: Array<out String>,
    grantResults: IntArray
  ) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)

    if (requestCode == REQUEST_PERMISSIONS) {
      if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
        // All permissions granted
        Toast.makeText(this, "Permissions granted", Toast.LENGTH_SHORT).show()
      } else {
        // Some permissions denied
        MaterialAlertDialogBuilder(this)
          .setTitle("Permissions Required")
          .setMessage("This app needs camera and storage permissions to function properly")
          .setPositiveButton("OK") { _, _ -> }
          .show()
      }
    }
  }

  override fun onDestroy() {
    super.onDestroy()
    mainScope.cancel()
    inferenceThread.close()
    ocrModel?.close()
  }
}