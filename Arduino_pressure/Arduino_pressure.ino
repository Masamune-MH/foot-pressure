int sensor_value[16]; // スケッチ冒頭の記述 

void setup() {

  Serial.begin(9600); // シリアルポート開始
}

void loop() {
  for(int i = 0; i < 16; i++) {
    sensor_value[i] = analogRead(i); 
    Serial.print(sensor_value[i]*5/1023);
    sensor_value[i] = map(sensor_value[i], 0, 1023, 0, 255); // 0〜255に変換
    Serial.print(sensor_value[i]);                // 数値として出力
    if(i < 15) Serial.print(",");                // カンマ区切り
  }
  
  // Serial.print('\n'); // Serial.printlnで改行されるため不要
  Serial.println();
  delay(1000); // 1秒待機 
}
