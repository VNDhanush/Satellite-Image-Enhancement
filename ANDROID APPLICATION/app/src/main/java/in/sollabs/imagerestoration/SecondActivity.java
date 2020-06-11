package in.sollabs.imagerestoration;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class SecondActivity extends AppCompatActivity {
    private Button denoisebutton,deblurbutton;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_second);
        denoisebutton = findViewById(R.id.button);
        deblurbutton = findViewById(R.id.button3);

        deblurbutton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(SecondActivity.this,WebDeblur.class);
                startActivity(i);
            }
        });

        denoisebutton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(SecondActivity.this,WebDenoise.class);
                startActivity(i);
            }
        });
    }
}
