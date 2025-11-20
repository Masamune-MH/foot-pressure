int sensor_value[16] ;//

void setup(){
  Serial.begin(9600);//serialport open
}

void loop(){
for(int i=0;i<16;i++){
  sensor_value[i] = analogRead(i);//Ai の値読み込み
  sensor_value[i] = map(sensor_value[i], 0, 1023, 0, 255);//1 バイト分に変換
  Serial.write(sensor_value[i]);//シリアルポートにデータ出力
}
Serial.print('¥n');
delay(3000);
}