package computer.gpu.tflitetest;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;


import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
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

    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    public Module loadModel() {
        try {
            Module module = Module.load(assetFilePath(this, "bert_traced_quant.pt"));
            return module;
        } catch (IOException ex) {
            Log.e(TAG, ex.getMessage());
            return null;
        }
    }

    public long runModel(Module model, Tensor input_ids, Tensor attention_mask, boolean verbose) {
        long startTime = System.nanoTime();
        IValue output = model.forward(IValue.from(input_ids), IValue.from(attention_mask));
        if(verbose){
            Log.i(TAG,
                    "Output == " + Arrays
                            .toString(
                                    output.toDictStringKey()
                                            .get("logits")
                                            .toTensor()
                                            .shape()
                            )
            );
        }
        long elapsedTime = System.nanoTime() - startTime;
        return elapsedTime;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Log.d(TAG, "========= Running =========");

        Module tflite = loadModel();
        long [] input_ids = {101, 1999, 2889, 1010, 1996, 3464, 1997, 2845, 17608, 6141, 2462, 1998, 2010,
                2155, 1006, 3272, 2005, 21219, 1998, 3814, 1007, 2024, 3603, 1012, 1996, 2376, 1997, 6141,
                1005, 1055, 2402, 2365, 1010, 17608, 16277, 21219, 24794, 16277, 1010, 6583, 11335, 4570,
                1996, 6893, 1997, 1996, 2466, 1012, 7257, 2530, 16881, 1010, 1037, 2402, 24665, 14031, 3089,
                20710, 18780, 2378, 2003, 2356, 2011, 2010, 2269, 1998, 1037, 2177, 1997, 2273, 2000, 4685,
                3894, 1012, 20710, 18780, 2378, 2038, 1037, 4432, 1998, 7939, 23709, 9623, 2028, 1997, 1996,
                2273, 2004, 1037, 3586, 12383, 1012, 2348, 2010, 2269, 3322, 14308, 2015, 2032, 2005, 2437,
                2107, 2019, 19238, 1010, 20710, 18780, 2378, 12197, 2004, 1996, 2158, 2003, 13303, 2648,
                1998, 7854, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0};
        long [] attention_mask = {1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};

        Tensor input_ids_tensor = Tensor.fromBlob(input_ids, new long[]{1, 128});
        Tensor attention_mask_tensor = Tensor.fromBlob(attention_mask, new long[]{1, 128});

        List<Long> times = new ArrayList<>();
        for(int i = 0; i < 500; i++) {
            times.add(runModel(tflite, input_ids_tensor, attention_mask_tensor, false));
        }
        double avg = 0.0, std = 0.0;
        long max = -1, min = 10000;
        for (Long time : times){
            if(time > max){
                max = time;
            }
            if(time < min){
                min = time;
            }
            avg += time / ((double) times.size());
        }
        for(Long time: times) {
            std += Math.pow(time - avg, 2);
        }
        std = Math.sqrt(std/(times.size()-1.0));

        Log.i(TAG, "Elapsed Time: " + avg/1_000_000 + "+-" + std/1_000_000 + " (ms)");
        Log.i(TAG, "Minimum: " + min/1_000_000.0 + " (ms)");
        Log.i(TAG, "Maximum: " + max/1_000_000.0 + " (ms)");
    }
}