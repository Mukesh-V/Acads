#include <U8g2lib.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

#include <WiFi.h>
#include "time.h"
#include "math.h"

#define PI 3.141592
#define up_down 33
#define sw 34
#define left_right 32
#define no_of_watch_faces 2
#define no_of_app 2
#define timeout 5
#define lift_threshold 12.0

const char* ssid     = "Wokwi-GUEST";
const char* password = "";

const char* ntpServer = "pool.ntp.org";
const long  gmtOffset_sec = 19800;
const int   daylightOffset_sec = 0;

U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* clock=*/ SCL, /* data=*/ SDA, /* reset=*/ U8X8_PIN_NONE); 
Adafruit_MPU6050 mpu;

char s_str[3];
char m_str[3];
char h_str[3];
char d_str[3];
char mn_str[3];
char y_str[3];

char temp_str[2];

int wake = 0;
int face_val = 0;
int chng_ud = 0;
int app_val = 0;
int chng_lr = 0;
int app_loop = 0;
long time_now;

struct tm timenow;

sensors_event_t a, g, temp;
int x_adj, y_adj;

int stpwch_run=0, stpwch_reset=0,stpwch_start=0;
long s_time_now;
int s_ms, s_s, s_m;
int s_c=0; 
char s_ms_str[3], s_s_str[3], s_m_str[3];
String s_msg;
String temp_unit;

void setup() {
  // put your setup code here, to run once:
  pinMode(up_down,INPUT);
  pinMode(left_right,INPUT);
  pinMode(sw,INPUT);
  if (!mpu.begin()) 
    Serial.println("Failed to find MPU6050 chip");
  mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
  u8g2.begin();   
  u8g2.setFontPosCenter();
  u8g2.setFont(u8g2_font_t0_11_tf);
  
  // Connect to Wi-Fi
  Serial.begin(115200);
  u8g2.clearBuffer(); 
  u8g2.setCursor(0,32);
  u8g2.drawStr(0,32,"Connecting to ");
  u8g2.drawStr(83,32,ssid);
  u8g2.sendBuffer();
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) 
  {
    delay(500);
  }
  u8g2.clearBuffer();
  u8g2.setCursor(0,32);
  u8g2.drawStr(25,32,"WiFi connected.");
  u8g2.sendBuffer();
  delay(500);
  // Init and get the time
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  if(!getLocalTime(&timenow))
  {
    Serial.println("Failed to obtain time");
    return;
  }
  
  //disconnect WiFi as it's no longer needed
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);
  u8g2.clearBuffer();
  u8g2.setCursor(0,32);
  u8g2.drawStr(20,32,"WiFi disconnected.");
  u8g2.sendBuffer();
  delay(500);
  u8g2.clear();
}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.println(wake);
  if(wake)
  {
    if(app_loop==0)
    {
      switch(chng_ud)
      {
        case 0:
          if( analogRead(up_down) < 1000 )
            chng_ud = 1;
          else if( analogRead(up_down) > 3000 )
            chng_ud = -1;
          break;
        case 1:
          if( analogRead(up_down) > 1000 )
          {
            face_val++;
            if( face_val > (no_of_watch_faces-1) )
              face_val = 0;
            chng_ud = 0;
            time_now = millis();
          }
          break;
        case -1:
          if( analogRead(up_down) < 3000 )
          {
            face_val--;
            if( face_val < 0 )
              face_val = no_of_watch_faces-1 ;
            chng_ud = 0;
            time_now = millis();
          }
          break;
      }
    }
    switch(chng_lr)
    {
      case 0:
        if( analogRead(left_right) < 1000 )
          chng_lr = -1;
        else if( analogRead(left_right) > 3000 )
          chng_lr = 1;
        break;
      case -1:
        if( analogRead(left_right) > 1000 )
        {
          app_val++;
          if( app_val > (no_of_app) )
            app_val = 0;
          chng_lr = 0;
          time_now = millis();
        }
        break;
      case 1:
        if( analogRead(left_right) < 3000 )
        {
          app_val--;
          if( app_val < 0 )
            app_val = no_of_app ;
          chng_lr = 0;
          time_now = millis();
        }
        break;
    }
  
    if(app_val == 0)
      app_loop = 0;
    else
      app_loop = 1;
    
    if(app_loop)
    {
      display_app(app_val);
      time_now = millis();
    }
    else
    {
      display_clock(face_val);
      
    }
    if(millis()-time_now > timeout*1000)
    {
      wake =0; 
      u8g2.clear();
    }
  }
  else
    lift_check();
}

void lift_check()
{
  mpu.getEvent(&a, &g, &temp);
  float a_mag = sqrt(pow(a.acceleration.x,2)+pow(a.acceleration.y,2)+pow(a.acceleration.z,2));
  if(a_mag > lift_threshold)
  {
    wake = 1;
    time_now = millis();
  }
}

void display_clock(int face_val)
{
  switch(face_val)
  {
    case 0:
      simple_time();
      break;
    case 1:
      date_time();
      break;
  }
    
}

void display_app(int app_val)
{
  switch(app_val)
  {
    case 1:
      timer();
      break;
    case 2:
      temperature();
      break;
  }
}

void date_time()
{
  if(!getLocalTime(&timenow))
    {
      Serial.println("Failed to obtain time");
      return;
    }
    strcpy(s_str, u8x8_u8toa(timenow.tm_sec, 2));
    strcpy(h_str, u8x8_u8toa(timenow.tm_hour, 2));
    strcpy(m_str, u8x8_u8toa(timenow.tm_min, 2));
    strcpy(d_str, u8x8_u8toa(timenow.tm_mday, 2));
    strcpy(mn_str, u8x8_u8toa(timenow.tm_mon, 2));
    strcpy(y_str, u8x8_u8toa(timenow.tm_year, 2));
    u8g2.clearBuffer();                   // clear the internal memory
    u8g2.setDrawColor(1);
    u8g2.setFont(u8g2_font_logisoso20_tn);   // choose a suitable font
    u8g2.drawStr(0,25,h_str);    // write something to the internal memory
    u8g2.drawStr(35,25,":");
    u8g2.drawStr(50,25,m_str);
    u8g2.drawStr(85,25,":");
    u8g2.drawStr(100,25,s_str);
    u8g2.setFont(u8g2_font_t0_11_tf);
    switch(timenow.tm_wday)
    {
      case 0:
        u8g2.drawButtonUTF8(20 , 55, U8G2_BTN_INV, 0,  2,  2, "SUN" );
        break;
      case 1:
        u8g2.drawButtonUTF8(20 , 55, U8G2_BTN_INV, 0,  2,  2, "MON" );
        break;
      case 2:
        u8g2.drawButtonUTF8(20 , 55, U8G2_BTN_INV, 0,  2,  2, "TUE" );
        break;
      case 3:
        u8g2.drawButtonUTF8(20 , 55, U8G2_BTN_INV, 0,  2,  2, "WED" );
        break;
      case 4:
        u8g2.drawButtonUTF8(20 , 55, U8G2_BTN_INV, 0,  2,  2, "THUR" );
        break;
      case 5:
        u8g2.drawButtonUTF8(20 , 55, U8G2_BTN_INV, 0,  2,  2, "FRI" );
        break;
      case 6:
        u8g2.drawButtonUTF8(20 , 55, U8G2_BTN_INV, 0,  2,  2, "SAT" );
        break;
         
    }
    String date;
    date += d_str;
    date += "/";
    date += mn_str;
    date += "/";
    date += y_str;
    u8g2.drawButtonUTF8(60 , 55, U8G2_BTN_INV, 0,  2,  2, date.c_str() );
    u8g2.sendBuffer();                    // transfer internal memory to the display
}

void simple_time()
{
  mpu.getEvent(&a, &g, &temp);
  if(!getLocalTime(&timenow))
    {
      Serial.println("Failed to obtain time");
      return;
    }
    strcpy(s_str, u8x8_u8toa(timenow.tm_sec, 2));
    strcpy(h_str, u8x8_u8toa(timenow.tm_hour, 2));
    strcpy(m_str, u8x8_u8toa(timenow.tm_min, 2));
    strcpy(d_str, u8x8_u8toa(timenow.tm_mday, 2));
    strcpy(mn_str, u8x8_u8toa(timenow.tm_mon, 2));
    strcpy(y_str, u8x8_u8toa(timenow.tm_year, 2));
    u8g2.clearBuffer();                   // clear the internal memory
    u8g2.setDrawColor(1);
    u8g2.drawBox(0,0,127,63);
    u8g2.setDrawColor(0);
    u8g2.setFont(u8g2_font_logisoso30_tn);
    u8g2.drawStr(20,31,h_str);
    u8g2.drawStr(55,31,":");
    u8g2.drawStr(65,31,m_str);
    u8g2.sendBuffer();                    // transfer internal memory to the display
}

void timer()
{
  u8g2.clearBuffer();
  u8g2.setDrawColor(1);
  u8g2.setFont(u8g2_font_logisoso20_tn); 
  if(digitalRead(34))
  {
    s_c = 1;
  }
  else
  {
    if(s_c)
    {
      s_c = 0;
      if(stpwch_reset)
      {
        if(stpwch_run)
        {
          
        }
        else
        {
          stpwch_reset=0;
        }
      }
      else
      {
        if(stpwch_run)
        {
          stpwch_run=0;
          stpwch_reset=1;
        }
        else
        {
          s_time_now = millis();
          stpwch_run=1;
        }
      }
    }
  }

  if(stpwch_run)
  {
    if(stpwch_reset)
    {
      
    }
    else
    {
      s_m=(millis()-s_time_now)/60000;
      s_s=((millis()-s_time_now)%60000)/1000;
      s_ms=((millis()-s_time_now)%1000)/10;
      s_msg="Press to  STOP";
    }
  }
  else
  {
    if(stpwch_reset)
    {
      s_msg="Press to RESET";
    }
    else
    {
      s_m=0;
      s_s=0;
      s_ms=0;
      s_msg="Press to START";
    }
  }
  
  strcpy(s_s_str, u8x8_u8toa(s_s, 2));
  strcpy(s_ms_str, u8x8_u8toa(s_ms, 2));
  strcpy(s_m_str, u8x8_u8toa(s_m, 2));
  u8g2.setFont(u8g2_font_logisoso20_tn);   // choose a suitable font
  u8g2.drawStr(0,25,s_m_str);    // write something to the internal memory
  u8g2.drawStr(35,25,":");
  u8g2.drawStr(50,25,s_s_str);
  u8g2.drawStr(85,25,":");
  u8g2.drawStr(100,25,s_ms_str);
  u8g2.setFont(u8g2_font_t0_11_tf);
  u8g2.drawStr(0,45,s_msg.c_str());
  u8g2.sendBuffer();
}

void temperature(){
  mpu.getEvent(&a, &g, &temp);
  temp_unit="degC";
  strcpy(temp_str, u8x8_u8toa((uint8_t)(temp.temperature), 2));
  u8g2.clearBuffer();                   // clear the internal memory
  u8g2.setDrawColor(1);
  u8g2.drawBox(0,0,127,63);
  u8g2.setDrawColor(0);
  u8g2.setFont(u8g2_font_logisoso30_tn);
  u8g2.drawStr(20,31,temp_str);
  u8g2.drawStr(20,31,temp_unit.c_str());
  u8g2.sendBuffer();
}