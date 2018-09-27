package com.caragiz_studioz.test.magicodev1;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

public class TensorFlowObjectDetectionAPIModel implements Classifier {
    private static final int MAX_RESULTS = 30;
    private String inputName;
    private int inputSize;

    private Vector<String> labels = new Vector<>();
    private int[] intValues;
    private byte[] byteValues;
    private float[] outputLocations;
    private float[] outputScores;
    private float[] outputClasses;
    private float[] outputNumDetections;
    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    public static Classifier create(final AssetManager assetManager,
                                    final String modelFilename,
                                    final String labelFilename,
                                    final int inputSize) throws IOException {
        final TensorFlowObjectDetectionAPIModel d =
                new TensorFlowObjectDetectionAPIModel();

        InputStream lablesInput = null;
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        lablesInput = assetManager.open(actualFilename);
        BufferedReader br = null;
        br = new BufferedReader(new InputStreamReader(lablesInput));
        String line;
        while ((line = br.readLine()) != null) {
            d.labels.add(line);
            Log.i("Label Found ", line);
        }
        br.close();

        d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);
        final Graph g = d.inferenceInterface.graph();
        d.inputName = "image_tensor";

        final Operation inputOp = g.operation(d.inputName);
        if (inputOp == null)
            throw new RuntimeException("Failed to find inputNode '" + d.inputName + "'");
        d.inputSize = inputSize;

        final Operation outputOp1 = g.operation("detection_scores");
        if (outputOp1 == null)
            throw new RuntimeException("Failed to find inputNode '" + d.inputName + "'");

        final Operation outputOp2 = g.operation("detection_boxes");
        if (outputOp2 == null)
            throw new RuntimeException("Failed to find node detection_boxes");

        final Operation outputOp3 = g.operation("detection_classes");
        if (outputOp3 == null)
            throw new RuntimeException("Failed to find node detection_classes");

        d.outputNames = new String[]{"detection_boxes", "detection_scores", "detection_classes",
                "num_detections"};
        d.intValues = new int[d.inputSize * d.inputSize];
        d.byteValues = new byte[d.inputSize * d.inputSize *3];
        d.outputScores = new float[MAX_RESULTS];
        d.outputLocations = new float[MAX_RESULTS * 4];
        d.outputClasses = new float[MAX_RESULTS];
        d.outputNumDetections = new float[1];

        return d;
    }

    private TensorFlowObjectDetectionAPIModel() {
    }

    @Override
    public List<Recognition> recognizeImage(Bitmap bitmap) {
        Trace.beginSection("recognizeImage");
        Trace.beginSection("preprocessBitmap");
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0
                , bitmap.getWidth(), bitmap.getHeight());
        try {
            for (int i = 0; i < intValues.length; ++i) {
                byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
                byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
                byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
            }
        } catch (ArrayIndexOutOfBoundsException e) {
        }

        Trace.endSection();

        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, byteValues, 1, inputSize, inputSize, 3);
        Trace.endSection();

        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        Trace.beginSection("fetch");
        outputLocations = new float[MAX_RESULTS * 4];
        outputScores = new float[MAX_RESULTS];
        outputClasses = new float[MAX_RESULTS];
        outputNumDetections = new float[1];
        inferenceInterface.fetch(outputNames[0], outputLocations);
        inferenceInterface.fetch(outputNames[1], outputScores);
        inferenceInterface.fetch(outputNames[2], outputClasses);
        inferenceInterface.fetch(outputNames[3], outputNumDetections);
        Trace.endSection();

        final PriorityQueue<Recognition> pq = new PriorityQueue<>(1,
                new Comparator<Recognition>() {
                    @Override
                    public int compare(Recognition o1, Recognition o2) {
                        return Float.compare(o2.getConfidence(), o1.getConfidence());
                    }
                });

        for (int i = 0; i < outputScores.length; ++i) {
            final RectF detection = new RectF(
                    outputLocations[4 * i + 1] * inputSize,
                    outputLocations[4 * i] * inputSize,
                    outputLocations[4 * i + 3] * inputSize,
                    outputLocations[4 * i + 2] * inputSize
            );
            pq.add(new Recognition("" + i, labels.get((int) outputClasses[i]),
                    outputScores[i], detection));
        }

        ArrayList<Recognition> recognitions = new ArrayList<>();
        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
            recognitions.add(pq.poll());
        }
        Trace.endSection();
        return recognitions;
    }

    @Override
    public void enableStatLogging(boolean debug) {
        this.logStats = debug;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
}
