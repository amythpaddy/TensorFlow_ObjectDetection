package com.caragiz_studioz.test.magicodev1;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

public class ActivityMain extends AppCompatActivity {

    private static final int TF_OF_API_INPUT_SIZE = 300;
    private static final String TF_OD_API_MODEL_FILE = "file:///android_asset/ssd_mobilenet_v1_android_export.pb";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }
}
