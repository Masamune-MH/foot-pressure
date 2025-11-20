int sensor_value[16];

void setup() {
  Serial.begin(9600); // serial port open
}

void loop() {
  for(int i = 0; i < 16; i++) {
    sensor_value[i] = analogRead(i);               // Ai の値読み込み
    sensor_value[i] = map(sensor_value[i], 0, 1023, 0, 255); // 1バイトに変換
    Serial.write(sensor_value[i]);                 // バイトとして送信
  }
  Serial.println(); // 改行
  delay(1000);
}
