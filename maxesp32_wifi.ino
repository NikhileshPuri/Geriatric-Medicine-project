#include <Wire.h>
#include "MAX30100_PulseOximeter.h"
#include <WiFi.h>
#include <WebServer.h>

#define SAMPLE_DELAY 100   // 100 ms = 10 Hz
#define SDA_PIN 25
#define SCL_PIN 26

// WiFi credentials
const char* ssid = "ESP32_HealthMonitor";
const char* password = "12345678";

PulseOximeter pox;
float hr = 0, spo2 = 0;
unsigned long lastSample = 0;

// Web server on port 80
WebServer server(80);

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Initialize I2C
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  Serial.println("Initializing MAX30100...");
  if (!pox.begin()) {
    Serial.println("MAX30100 not detected! Check wiring!");
    while (1);
  }
  pox.setIRLedCurrent(MAX30100_LED_CURR_7_6MA);
  Serial.println("MAX30100 initialized successfully.");

  // Start WiFi AP
  WiFi.softAP(ssid, password);
  Serial.println("WiFi AP started");
  Serial.print("Connect your phone to: ");
  Serial.println(WiFi.softAPIP());

  // Web server routes
  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.begin();
}

void loop() {
  pox.update();

  // Sample every SAMPLE_DELAY
  if (millis() - lastSample > SAMPLE_DELAY) {
    lastSample = millis();
    hr = pox.getHeartRate();
    spo2 = pox.getSpO2();
  }

  server.handleClient();
}

// Serve HTML dashboard
void handleRoot() {
  server.send(200, "text/html", R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <title>Health Monitor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: Arial, Helvetica, sans-serif; margin:0; padding:20px; background:#0d1117; color:#f0f6fc; }
    h2 { text-align:center; font-size:32px; color:#58a6ff; margin-bottom:20px; }
    .card { background:#161b22; padding:24px; margin:16px auto; border-radius:16px; box-shadow:0 4px 12px rgba(0,0,0,0.5); max-width:600px; }
    .val { font-size:28px; font-weight:700; display:block; margin-top:8px; }
    canvas { width:100%; height:200px; background:#1c1c1c; border-radius:12px; display:block; margin-top:16px; }
  </style>
</head>
<body>
<h2>ESP32 Health Dashboard</h2>

<div class="card">
  <b>Heart Rate & SpO2</b>
  <span class="val">HR: <span id="hrVal">0</span> bpm | SpO2: <span id="spo2Val">0</span>%</span>
  <canvas id="graph"></canvas>
</div>

<script>
const canvas = document.getElementById('graph');
const ctx = canvas.getContext('2d');
let dataHR = [], dataSpO2 = [], maxPoints = 50;

function drawGraph() {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  drawLine(dataHR, 'red');
  drawLine(dataSpO2, 'blue');
}

function drawLine(data, color) {
  ctx.strokeStyle = color;
  ctx.beginPath();
  for(let i=0;i<data.length;i++){
    let y = canvas.height - (data[i]/200)*canvas.height;
    if(i==0) ctx.moveTo(i*(canvas.width/maxPoints),y);
    else ctx.lineTo(i*(canvas.width/maxPoints),y);
  }
  ctx.stroke();
}

async function fetchData(){
  const response = await fetch('/data');
  const json = await response.json();

  document.getElementById('hrVal').innerText = json.hr.toFixed(1);
  document.getElementById('spo2Val').innerText = json.spo2.toFixed(1);

  if(dataHR.length>=maxPoints){ dataHR.shift(); dataSpO2.shift(); }
  dataHR.push(json.hr); dataSpO2.push(json.spo2);
  drawGraph();
}

fetchData();
setInterval(fetchData, 200);
</script>
</body>
</html>
  )rawliteral");
}

// Serve sensor data as JSON
void handleData() {
  String json = "{";
  json += "\"hr\":" + String(hr) + ",";
  json += "\"spo2\":" + String(spo2);
  json += "}";
  server.send(200, "application/json", json);
}
