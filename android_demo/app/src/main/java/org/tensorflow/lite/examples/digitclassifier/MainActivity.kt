package org.tensorflow.lite.examples.digitclassifier

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.divyanshu.draw.widget.DrawView
import kotlinx.android.synthetic.main.activity_main.*
import java.lang.Exception

class MainActivity : AppCompatActivity() {

  private var drawView: DrawView? = null
  private var clearButton: Button? = null
  private var resetButton: Button? = null
  private var predictedTextView: TextView? = null
  private var predictedImageView: ImageView? = null
  private var spinner: Spinner? = null
  private var digitClassifier = DigitClassifier(this)
  private var languageList = ArrayList<String>()
  private var supportSet = ArrayList<Bitmap>()
  private var currentTest: Bitmap? = null

  @SuppressLint("ClickableViewAccessibility")
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    //Fetch initial data from assets
    get_lang_list()
    supportSet = get_lang_data(languageList[0])

    //Set Up Spinner
    spinner = findViewById(R.id.language_list)
    val aa = ArrayAdapter(this, android.R.layout.simple_spinner_item, languageList)
    aa.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
    spinner!!.setAdapter(aa)

    //set up fetcher
    spinner?.onItemSelectedListener = object : AdapterView.OnItemSelectedListener{
      override fun onNothingSelected(parent: AdapterView<*>?) {

      }

      override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
        val selectedItem = parent!!.getItemAtPosition(position).toString()
        supportSet = get_lang_data(selectedItem)
        try {
          startNewTest()
        }catch(e:Exception){}
      }

    }

    //initial test
    startNewTest()

    // Setup view instances
    drawView = findViewById(R.id.draw_view)
    drawView?.setStrokeWidth(50.0f)
    drawView?.setColor(Color.BLACK)
    drawView?.setBackgroundColor(Color.WHITE)
    clearButton = findViewById(R.id.clear_button)
    resetButton = findViewById(R.id.new_task_button)
    predictedTextView = findViewById(R.id.predicted_text)
    predictedImageView = findViewById(R.id.predicted_image)


    // Setup clear drawing button
    clearButton?.setOnClickListener {
      drawView?.clearCanvas()
      predictedTextView?.text = getString(R.string.prediction_text_placeholder)
    }

    // Setup reset task button
    resetButton?.setOnClickListener {
      startNewTest()
      drawView?.clearCanvas()
      predictedTextView?.text = getString(R.string.prediction_text_placeholder)
    }

    // Setup classification trigger so that it classify after every stroke drew
    drawView?.setOnTouchListener { _, event ->
      // As we have interrupted DrawView's touch event,
      // we first need to pass touch events through to the instance for the drawing to show up
      drawView?.onTouchEvent(event)

      // Then if user finished a touch event, run classification
      if (event.action == MotionEvent.ACTION_UP) {
        classifyDrawing(arrayListOf(currentTest!!))
      }

      true
    }

    // Setup digit classifier
    digitClassifier
      .initialize()
      .addOnFailureListener { e -> Log.e(TAG, "Error to setting up digit classifier.", e) }
  }

  //get Language List
  private fun get_lang_list(){
    val f = assets.list("images_background")
    languageList = ArrayList()
    for(f1 in f)
      languageList.add(f1)
    Log.d("Language List",languageList.toString())
  }

  //get language data
  private fun get_lang_data(language:String):ArrayList<Bitmap> {
    val path =  "images_background/${language}"
    val result = ArrayList<Bitmap>()
    for(char in assets.list(path))
    {
      val inps = assets.open(path + "/${char}/${assets.list(path + "/${char}")[0]}")
      result.add(BitmapFactory.decodeStream(inps))
    }
    return result
  }

  //get a test Sample
  private fun getSample(tray:ArrayList<Bitmap>):Bitmap{
    return tray.shuffled().take(1)[0]
  }

  //set up test env
  private fun startNewTest(){
    val testImage = getSample(supportSet)
    predictedImageView?.setImageBitmap(testImage)
    currentTest = testImage
  }

  override fun onDestroy() {
    digitClassifier.close()
    super.onDestroy()
  }

  private fun classifyDrawing(supportSet: ArrayList<Bitmap>) {
    val bitmap = drawView?.getBitmap()

    if ((bitmap != null) && (digitClassifier.isInitialized)) {
      digitClassifier
        .classifyAsync(bitmap,supportSet)
        .addOnSuccessListener { resultText ->
          run {
            val acc = resultText.split("-")[0].toFloat()
            fun Double.format(digits: Int) = "%.${digits}f".format(this)
            if(acc > 0.72f)
            {
              startNewTest()
              predictedTextView?.text = "Well Done"
              drawView?.clearCanvas()
            }else
            {
              predictedTextView?.text = "Only ${(acc*100).toDouble().format(2)}% accuracy"
            }
          }
        }
        .addOnFailureListener { e ->
          predictedTextView?.text = getString(
            R.string.classification_error_message,
            e.localizedMessage
          )
          Log.e(TAG, "Error classifying drawing.", e)
        }
    }
  }

  companion object {
    private const val TAG = "MainActivity"
  }
}
