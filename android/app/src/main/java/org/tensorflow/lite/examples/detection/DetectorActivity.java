/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.SparseArray;
import android.util.TypedValue;
import android.widget.Toast;

import com.google.android.gms.vision.Detector;
import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.Tracker;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;
import com.google.android.gms.vision.face.LargestFaceFocusingProcessor;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteFaceRecognitionAPIModel;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    // Configuration values for the prepackaged SSD model.
    private static final int TF_OD_API_INPUT_SIZE = 160;
    private static final boolean TF_OD_API_IS_QUANTIZED = true;
    private static final String TF_OD_API_MODEL_FILE = "facenet.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(320, 240);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private Classifier detector;

    private FaceDetector faceDetector;

    Toast distanceToast;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {

        distanceToast = new Toast(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            detector =
                    TFLiteFaceRecognitionAPIModel.create(getAssets());
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e("Exception initializing classifier!", e);
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);
    }

    private void showDistance(String distance) {
        if (distanceToast != null && distanceToast.getView() != null && distanceToast.getView().isShown()) {
            distanceToast.setText(distance);
        } else {
            distanceToast = Toast.makeText(this, distance, Toast.LENGTH_SHORT);
            distanceToast.show();
        }
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

//        faceDetector = new FaceDetector.Builder(this)
//                .setProminentFaceOnly(true)
//                .build();
//
//        faceDetector.setProcessor(
//                new LargestFaceFocusingProcessor(
//                        faceDetector,
//                        new FaceTracker()));

        runInBackground(
                () -> {
                    LOGGER.i("Running detection on image " + currTimestamp);
                    final long startTime = SystemClock.uptimeMillis();
                    final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);

                    showDistance(String.valueOf(((TFLiteFaceRecognitionAPIModel) detector).getDistance()));

//                    Frame currentFrame = new Frame.Builder()
//                            .setBitmap(croppedBitmap)
//                            .build();
//
//                    SparseArray<Face> detectedFace = faceDetector.detect(currentFrame);

                    lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                    cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                    final Canvas canvas1 = new Canvas(cropCopyBitmap);
                    final Paint paint = new Paint();
                    paint.setColor(Color.RED);
                    paint.setStyle(Style.STROKE);
                    paint.setStrokeWidth(2.0f);

                    float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                    switch (MODE) {
                        case TF_OD_API:
                            minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                            break;
                    }

                    final List<Classifier.Recognition> mappedRecognitions =
                            new LinkedList<Classifier.Recognition>();

//                    for (int i = 0; i < detectedFace.size(); i++) {
//                        Face f = detectedFace.get(i);
//
//                        if (f != null && f.getPosition() != null) {
//                            Log.d("ITERATION", "LEFT: " + f.getPosition().x);
//
//                            final RectF location = new RectF(f.getPosition().x, f.getPosition().y,
//                                    f.getPosition().x + f.getWidth(), f.getPosition().y + f.getHeight());
//
//                            canvas1.drawRect(location, paint);
//
//                            cropToFrameTransform.mapRect(location);
//
//                            Classifier.Recognition recognition = new Classifier.Recognition(
//                                    String.valueOf(f.getId()), "Face", 1f, location);
//                            mappedRecognitions.add(recognition);
//                        }
//                    }

                    for (final Classifier.Recognition result : results) {
                        final RectF location = result.getLocation();
                        if (location != null && result.getConfidence() >= minimumConfidence) {
                            canvas.drawRect(location, paint);

                            cropToFrameTransform.mapRect(location);

                            result.setLocation(location);
                            mappedRecognitions.add(result);
                        }
                    }
                    computingDetection = false;
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

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }

    class FaceTracker extends Tracker<Face> {

        public FaceTracker() {
        }

        public void onNewItem(int var1, Face var2) {
        }

        public void onUpdate(Detector.Detections<Face> var1, Face var2) {
        }

        public void onMissing(Detector.Detections<Face> var1) {
        }

        public void onDone() {
        }
    }
}