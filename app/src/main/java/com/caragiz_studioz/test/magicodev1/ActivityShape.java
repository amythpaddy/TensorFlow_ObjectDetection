package com.caragiz_studioz.test.magicodev1;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.TextView;
import android.widget.Toast;

import com.caragiz_studioz.test.magicodev1.env.BorderedText;
import com.caragiz_studioz.test.magicodev1.env.ImageUtils;
import com.caragiz_studioz.test.magicodev1.fromOrigin.CameraActivity;
import com.caragiz_studioz.test.magicodev1.tracking.MultiBoxTracker;

import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class ActivityShape extends CameraActivity {

    private static final int TF_OF_API_INPUT_SIZE = 300;
//    private static final String TF_OD_API_MODEL_FILE = "file:///android_asset/graph.pb";
//    private static final String TF_OD_API_MODEL_FILE = "file:///android_asset/magico17aug.pb";
    private static final String TF_OD_API_MODEL_FILE = "file:///android_asset/magico_shape_graph.pb";
//    private static final String TF_OD_API_MODEL_FILE = "file:///android_asset/mumbai_demo_test.pb";
//    private static final String TF_OD_API_LABELS = "file:///android_asset/labels.txt";
//    private static final String TF_OD_API_LABELS = "file:///android_asset/magico_labels.txt";
    private static final String TF_OD_API_LABELS = "file:///android_asset/magico_shape_labels.txt";
//    private static final String TF_OD_API_LABELS = "file:///android_asset/magico_dev_labels.txt";

    private enum DetectorMode {
        TF_OD_API, MULTIBOX, YOLO;
    }

    private static final DetectorMode MODE = DetectorMode.YOLO;

    private static final float MINIMUM_CONFIDENCE = 0.1f;
    private static final boolean MAINTAIN_ASPECT = MODE == DetectorMode.YOLO;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE = 10;

    private static final String YOLO_MODEL_FILE = "file:///android_asset/yolov2_tiny.pb";
    private static final int YOLO_INPUT_SIZE = 360;
    private static final String YOLO_INPUT_NAME = "input";
    private static final String YOLO_OUTPUT_NAMES = "output";
    private static final int YOLO_BLOCK_SIZE = 32;

    private Integer sensorOrientation;
    private Classifier detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;
    private int width = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private byte[] luminanceCopy;
    private BorderedText borderedText;

    private OverlayView trackingOverlay;
    private MultiBoxTracker tracker;

    private TextView displayScreen1;
    private TextView displayScreen2;
    private TextView displayScreen3;
    private TextView displayScreen4;
    private int randNum1;
    private int randNum2;
    private int randNum3;
    private int randNum4;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        displayScreen1 = findViewById(R.id.display_screen1);
        displayScreen2 = findViewById(R.id.display_screen2);
        displayScreen3 = findViewById(R.id.display_screen3);
        displayScreen4 = findViewById(R.id.display_screen4);

        DisplayMetrics display = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(display);
        width = display.widthPixels;
        generateNumber();
    }

    private void generateNumber() {
        Random r = new Random();
        randNum1 = r.nextInt(10) + 1;
        displayScreen1.setText(randNum1+"");

        randNum2 = r.nextInt(10) + 1;
        displayScreen2.setText(randNum2+"");

        randNum3 = r.nextInt(10) + 1;
        displayScreen3.setText(randNum3+"");

        randNum4 = r.nextInt(10) + 1;
        displayScreen4.setText(randNum4+"");
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        byte[] originalLuminance = getLuminance();

        tracker.onFrame(
                previewWidth,
                previewHeight,
                getLuminanceStride(),
                sensorOrientation,
                originalLuminance,
                timestamp
        );
        trackingOverlay.postInvalidate();

        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        rgbFrameBitmap.setPixels(getRgbBytes(), 0,
                previewWidth, 0, 0,
                previewWidth, previewHeight);

        if (luminanceCopy == null) {
            luminanceCopy = new byte[originalLuminance.length];
        }
        System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(new Runnable() {
            @Override
            public void run() {
                Log.i("Running Detecttion at", currTimestamp + "");
                final long startTime
                        = SystemClock.uptimeMillis();
                final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                final Canvas canvas = new Canvas(cropCopyBitmap);
                final Paint paint = new Paint();
                paint.setColor(Color.RED);
                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeWidth(2.0f);

                float minimumConfidence = MINIMUM_CONFIDENCE;
                final List<Classifier.Recognition> mappedRecognintions = new LinkedList<>();

                for (final Classifier.Recognition result : results) {
                    final RectF location = result.getLocation();
//                    location.right = width - location.right;
//                    location.left = width - location.left;
                    if (location != null && result.getConfidence() >= minimumConfidence) {
                        canvas.drawRect(location, paint);

                        cropToFrameTransform.mapRect(location);
                        result.setLocation(location);
                        mappedRecognintions.add(result);

                    }
                    else{
                        canvas.drawColor(Color.TRANSPARENT);
                    }
                }
                tracker.trackResults(mappedRecognintions, luminanceCopy, currTimestamp);
                trackingOverlay.postInvalidate();
                checkAns(mappedRecognintions);

                computingDetection = false;
            }
        });

    }

    private void checkAns(List<Classifier.Recognition> mappedRecognintions) {
        int temp = 0;
        Iterator<Classifier.Recognition> recognitions = mappedRecognintions.iterator();
        while (recognitions.hasNext()) {
            String id =recognitions.next().getTitle();
            switch (id) {
                case "one":
                    temp = temp + 1;
                    break;
                case "two":
                    temp = temp + 2;
                    break;
                case "five":
                case "five_alt":
                    temp += 5;
                    break;
                default:
                    temp += 0;
            }
        }

        if (temp == randNum1)
        {
            randNum1 = 0;
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    displayScreen1.setText("YAAY!!!");
                }
            });

        }else if (temp == randNum2)
        {
            randNum2 = 0;
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    displayScreen2.setText("YAAY!!!");
                }
            });

        }else if (temp == randNum3)
        {
            randNum3 = 0;
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    displayScreen3.setText("YAAY!!!");
                }
            });

        }else if (temp == randNum4)
        {
            randNum4 = 0;
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    displayScreen4.setText("YAAY!!!");
                }
            });

        }
    }

    @Override
    protected void onPreviewSizeChosen(Size size, int rotation) {
        final float textSize = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP,
                TEXT_SIZE, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSize);
        borderedText.setTypeface(Typeface.MONOSPACE);

        int cropSize = TF_OF_API_INPUT_SIZE;
        if (MODE == DetectorMode.YOLO) {
            detector =
                    TensorFlowYoloDetector.create(
                            getAssets(),
                            YOLO_MODEL_FILE,
                            YOLO_INPUT_SIZE,
                            YOLO_INPUT_NAME,
                            YOLO_OUTPUT_NAMES,
                            YOLO_BLOCK_SIZE);
            cropSize = YOLO_INPUT_SIZE;
        }
        else{
            try {
                detector = TensorFlowObjectDetectionAPIModel.create(
                        getAssets(),
                        TF_OD_API_MODEL_FILE,
                        TF_OD_API_LABELS,
                        TF_OF_API_INPUT_SIZE
                );
            } catch (IOException e) {
                e.printStackTrace();
                Toast.makeText(this, "Classifier cannot be initialized", Toast.LENGTH_SHORT).show();
                finish();
            }
        }

        previewHeight = size.getHeight();
        previewWidth = size.getWidth();

        sensorOrientation = rotation - getScreenOrientation();
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);

        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                cropSize, cropSize,
                sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = findViewById(R.id.tracking_overlay);
        tracker = new MultiBoxTracker(this);
        trackingOverlay.addCallback(new OverlayView.DrawCallback() {
            @Override
            public void drawCallback(Canvas canvas) {
                tracker.draw(canvas);
            }
        });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }


}
