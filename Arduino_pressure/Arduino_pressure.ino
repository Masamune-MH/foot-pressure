// センサー値を格納する配列
int sensor_value[16]; 

void setup() {
  Serial.begin(9600); // シリアルポート開始
  
}

void loop() {
  // 1. センサーデータを読み取って送信
  for(int i = 0; i < 16; i++) {
    // Arduino Megaなどピンが多いボードでない場合、A6以降は読み取れない可能性があります
    sensor_value[i] = analogRead(i); 

    // 0〜1023 のアナログ値を 0〜255 (1byte) に変換
    int byte_val = map(sensor_value[i], 0, 1023, 0, 255);

    // 【重要】数値(バイナリ)として送信
    Serial.write(byte_val); 
  }
  
  // 2. 同期用の改行コードを送信
  Serial.write('\n'); 

  // 3. 待機 (Pythonの描画速度に合わせて調整)
  delay(200); 
}