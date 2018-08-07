package com.caragiz_studioz.test.magicodev1;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.os.Handler;
import android.os.SystemClock;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Surface;
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

public class ActivityMain extends CameraActivity {

    private static final int TF_OF_API_INPUT_SIZE = 300;
    private static final String TF_OD_API_MODEL_FILE = "file:///android_asset/graph.pb";
    private static final String TF_OD_API_LABELS = "file:///android_asset/labels.txt";

    private enum DetectorMode {
        TF_OD_API, MULTIBOX, YOLO;
    }

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;

    private static final float MINIMUM_CONFIDENCE = 0.6f;
    private static final boolean MAINTAIN_ASPECT = MODE == DetectorMode.YOLO;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE = 10;

    private Integer sensorOrientation;
    private Classifier detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private byte[] luminanceCopy;
    private BorderedText borderedText;

    private OverlayView trackingOverlay;
    private MultiBoxTracker tracker;

    private TextView displayScreen;
    private int randNum;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        displayScreen = findViewById(R.id.display_screen);
        generateNumber();
    }

    private void generateNumber() {
        Random r = new Random();
        randNum = r.nextInt(10) + 1;
        displayScreen.setText(randNum+"");
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
                    if (location != null && result.getConfidence() >= minimumConfidence) {
                        canvas.drawRect(location, paint);

                        cropToFrameTransform.mapRect(location);
                        result.setLocation(location);
                        mappedRecognintions.add(result);

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

        if (temp == randNum)
        {
            displayScreen.setText("YOU ARE RIGHT");
            new Handler().postDelayed(new Runnable() {
                @Override
                public void run() {
                    generateNumber();
                }
            },1500);
        }
    }

    @Override
    protected void onPreviewSizeChosen(Size size, int rotation) {
        final float textSize = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP,
                TEXT_SIZE, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSize);
        borderedText.setTypeface(Typeface.MONOSPACE);

        int cropSize = TF_OF_API_INPUT_SIZE;
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
