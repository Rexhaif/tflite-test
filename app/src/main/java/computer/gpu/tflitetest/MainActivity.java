package computer.gpu.tflitetest;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;


public class MainActivity extends AppCompatActivity {
    public static final String TAG = "==== MAIN ====";

    public MappedByteBuffer loadModelFile(AssetManager assetManager) throws IOException {
        try (AssetFileDescriptor fileDescriptor = assetManager.openFd("l2_h128.tflite");
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    public Interpreter loadModel() {
        try {
            ByteBuffer buffer = loadModelFile(this.getApplicationContext().getAssets());
            Interpreter.Options opt = new Interpreter.Options();
            opt.setNumThreads(1);
            Interpreter tflite = new Interpreter(buffer, opt);
            Log.v(TAG, "TFLite model loaded.");
            return tflite;
        } catch (IOException ex) {
            Log.e(TAG, ex.getMessage());
            return null;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Log.d("Main", "========= Running =========");

        Interpreter tflite = loadModel();
        tflite.allocateTensors();
        int [][] input_ids = {{101, 1999, 2889, 1010, 1996, 3464, 1997, 2845, 17608, 6141, 2462, 1998, 2010,
                2155, 1006, 3272, 2005, 21219, 1998, 3814, 1007, 2024, 3603, 1012, 1996, 2376, 1997, 6141,
                1005, 1055, 2402, 2365, 1010, 17608, 16277, 21219, 24794, 16277, 1010, 6583, 11335, 4570,
                1996, 6893, 1997, 1996, 2466, 1012, 7257, 2530, 16881, 1010, 1037, 2402, 24665, 14031, 3089,
                20710, 18780, 2378, 2003, 2356, 2011, 2010, 2269, 1998, 1037, 2177, 1997, 2273, 2000, 4685,
                3894, 1012, 20710, 18780, 2378, 2038, 1037, 4432, 1998, 7939, 23709, 9623, 2028, 1997, 1996,
                2273, 2004, 1037, 3586, 12383, 1012, 2348, 2010, 2269, 3322, 14308, 2015, 2032, 2005, 2437,
                2107, 2019, 19238, 1010, 20710, 18780, 2378, 12197, 2004, 1996, 2158, 2003, 13303, 2648,
                1998, 7854, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0}};
        int [][] attention_mask = {{1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}};

        float [][][] outputs = new float[1][128][30522];
        Object[] inputs = {attention_mask, input_ids};
        HashMap<Integer, Object> outputsMap = new HashMap<>();
        outputsMap.put(0, outputs);
        long startTime = System.nanoTime();
        tflite.runForMultipleInputsOutputs(inputs, outputsMap);
        long elapsedTime = System.nanoTime() - startTime;
        Log.i(TAG, "Elapsed Time: " + elapsedTime/1_000_000 + " (ms)");
    }
}