#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

#define LED_PIN D5
#define CRASH_THRESHOLD 35
#define SAMPLE_DELAY 50
#define LED_ON_TIME 500

// WiFi credentials
const char* ssid = "CrashDetector";
const char* password = "12345678";

Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);
ESP8266WebServer server(80);

float ax, ay, az;
bool crashDetected = false;
unsigned long ledTimer = 0;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  if (!accel.begin()) {
    Serial.println("ADXL345 not detected!");
    while (1);
  }
  accel.setRange(ADXL345_RANGE_16_G);

  // Start WiFi in AP mode
  WiFi.softAP(ssid, password);
  Serial.print("Connect to WiFi: ");
  Serial.println(ssid);
  Serial.print("IP Address: ");
  Serial.println(WiFi.softAPIP());

  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.begin();
}

void loop() {
  sensors_event_t event;
  accel.getEvent(&event);

  ax = abs(event.acceleration.x);
  ay = abs(event.acceleration.y);
  az = abs(event.acceleration.z);

  crashDetected = (ax > CRASH_THRESHOLD || ay > CRASH_THRESHOLD || az > CRASH_THRESHOLD);

  if (crashDetected) {
    digitalWrite(LED_PIN, HIGH);
    ledTimer = millis() + LED_ON_TIME;
  }

  if (millis() > ledTimer) {
    digitalWrite(LED_PIN, LOW);
  }

  server.handleClient();
  delay(SAMPLE_DELAY);
}

// Serve the HTML page
void handleRoot() {
  server.send(200, "text/html", R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Crash Detector Dashboard</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;background:#0d1117;color:#f0f6fc;padding:20px;margin:0}
h2{text-align:center;font-size:32px;margin-bottom:20px;color:#58a6ff}
.card{background:#161b22;padding:24px;margin:16px auto;border-radius:16px;box-shadow:0 4px 12px rgba(0,0,0,0.5);max-width:600px}
.val{font-size:28px;font-weight:700;display:block;margin-top:8px}
.status-card{background:#1c1c1c;padding:24px;border-radius:16px;text-align:center;font-size:24px;margin-top:20px;font-weight:bold}
.status-normal{color:#3fb950}
.status-crash{color:#f85149}
canvas{width:100%;height:200px;background:#1c1c1c;border-radius:12px;display:block;margin-top:16px}
</style>
</head>
<body>
<h2>Crash Detector Dashboard</h2>

<div class="card status-card status-normal" id="status">Loading...</div>

<div class="card">
  <b>Accelerometer Values</b>
  <span class="val">X: <span id="xVal">0</span> m/s² | Y: <span id="yVal">0</span> m/s² | Z: <span id="zVal">0</span> m/s²</span>
  <canvas id="graph"></canvas>
</div>

<script>
const canvas=document.getElementById('graph');
const ctx=canvas.getContext('2d');
const statusCard=document.getElementById('status');
let dataX=[],dataY=[],dataZ=[],maxPoints=50;

function drawGraph(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  drawLine(dataX,'red');
  drawLine(dataY,'green');
  drawLine(dataZ,'blue');
}

function drawLine(data,color){
  ctx.strokeStyle=color;
  ctx.beginPath();
  for(let i=0;i<data.length;i++){
    let y=canvas.height-(data[i]/50)*canvas.height;
    if(i==0) ctx.moveTo(i*(canvas.width/maxPoints),y);
    else ctx.lineTo(i*(canvas.width/maxPoints),y);
  }
  ctx.stroke();
}

async function fetchData(){
  const response=await fetch('/data');
  const json=await response.json();

  // Update status card
  if(json.crash){
    statusCard.innerText='CRASH DETECTED!';
    statusCard.classList.add('status-crash');
    statusCard.classList.remove('status-normal');
  } else {
    statusCard.innerText='Normal';
    statusCard.classList.add('status-normal');
    statusCard.classList.remove('status-crash');
  }

  // Update accelerometer values
  document.getElementById('xVal').innerText=json.ax.toFixed(1);
  document.getElementById('yVal').innerText=json.ay.toFixed(1);
  document.getElementById('zVal').innerText=json.az.toFixed(1);

  // Update graph
  if(dataX.length>=maxPoints){ dataX.shift(); dataY.shift(); dataZ.shift(); }
  dataX.push(json.ax); dataY.push(json.ay); dataZ.push(json.az);
  drawGraph();
}

fetchData();
setInterval(fetchData,100);
</script>
</body>
</html>
)rawliteral");
}




// Serve acceleration data as JSON
void handleData() {
  String json = "{";
  json += "\"ax\":" + String(ax) + ",";
  json += "\"ay\":" + String(ay) + ",";
  json += "\"az\":" + String(az) + ",";
  json += "\"crash\":" + String(crashDetected ? "true":"false");
  json += "}";
  server.send(200, "application/json", json);
}
