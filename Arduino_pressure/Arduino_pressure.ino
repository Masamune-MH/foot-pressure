int sensor_value[16]; // スケッチ冒頭の記述 

void setup() {
  Serial.begin(9600); // 9600 bpsでシリアルポートを開く 
}

void loop() {
  for(int i=0; i<16; i++){
    sensor_value[i] = analogRead(i); // A1の読み込み 
    
    // センサー値を0-1023から0-255に変換しているが、
    // ここで8ビットに圧縮する必要がなければそのまま使用する方が良い。
    // 圧縮する場合は、この行は残す。
    sensor_value[i] = map(sensor_value[i], 0, 1023, 0, 255); 
    
    // ⭐︎修正ポイント: Serial.write()からSerial.print/printlnに変更
    // 値を数値（ASCII）としてシリアルモニタに出力
    Serial.print("Sensor "); 
    Serial.print(i);
    Serial.print(": ");
    Serial.println(sensor_value[i]); // 値を出力し、改行
  }
  
  // Serial.print('\n'); // Serial.printlnで改行されるため不要
  delay(1000); // 1秒待機 
}
