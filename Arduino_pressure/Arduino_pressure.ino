int sensor_value[16];

void setup(){
  Serial.begin(9600);
}

void loop(){
  for(int i=0; i<16; i++){
    sensor_value[i] = analogRead(i); // 0～1023の値をそのまま取得

    // ★ここが変更点★
    // map関数は削除します
    // 1023は2バイト(16bit)必要なので、上位8bitと下位8bitに分けて送ります
    
    Serial.write(sensor_value[i] >> 8);   // 上位8ビットを送信
    Serial.write(sensor_value[i] & 0xFF); // 下位8ビットを送信
  }
  Serial.print('\n'); // 区切りの改行
  delay(1000);
}