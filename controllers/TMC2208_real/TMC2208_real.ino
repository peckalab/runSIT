/*Example sketch to control a stepper motor with A4988 stepper motor driver and Arduino without a library. More info: https://www.makerguides.com */

// Define stepper motor connections and steps per revolution:
#define M1dirPin 4
#define M1stepPin 5
#define M2dirPin 8
#define M2stepPin 9

#define enablePin 6
#define stepsPerRevolution 1600 

int dir = 1;   // rotation direction 1 or -1
int steps = 0; // number of steps to rotate
int spd = 20;  // speed in some units
int ind = 0;
unsigned int latency = 1; // speed as a latency between motor commands
String command; // input command

void setup() {
  // Declare pins as output:
  pinMode(M1dirPin, OUTPUT);
  pinMode(M1stepPin, OUTPUT);
  pinMode(M2dirPin, OUTPUT);
  pinMode(M2stepPin, OUTPUT);
  pinMode(enablePin, OUTPUT);

  Serial.begin(9600);
}


void rotate(int dir, int steps, int spd) {
  digitalWrite(enablePin, LOW);
  
  // set the direction
  if (dir > 0) {
    digitalWrite(M1dirPin, HIGH);
    digitalWrite(M2dirPin, HIGH);
  } else {
    digitalWrite(M1dirPin, LOW);
    digitalWrite(M2dirPin, LOW);
  }

  // compute speed as a latency
  latency = 100000 / spd;
  Serial.println(latency);
  
  // Spin the motors
  for (int i = 0; i < 8 * steps; i++) {
    digitalWrite(M1stepPin, HIGH);
    digitalWrite(M2stepPin, HIGH);
    delayMicroseconds(latency);
    digitalWrite(M1stepPin, LOW);
    digitalWrite(M2stepPin, LOW);
    delayMicroseconds(latency);
  }

  digitalWrite(enablePin, HIGH);
}


void loop() {
  if (Serial.available())  {
    char c = Serial.read();  //gets one byte from serial buffer
    
    if (c == '\n') {
      Serial.println(command); //prints string to serial port out
      
      // rotation direction
      String dirCmd = command.substring(0, 1);
      if (dirCmd.equals("+")) {
        dir = 1;
      } else {
        dir = -1;
      }

      // number of steps to rotate
      ind = command.indexOf(':');
      steps = command.substring(1, ind).toInt();

      // rotation speed
      spd = command.substring(ind + 1, command.length()).toInt();

      rotate(dir, steps, spd);
      command = ""; //clears variable for new input

      // output of the current position
      //Serial.println(stepper1.currentPosition());
    }  
    else {     
      command += c; //makes the string readString
    }
  }
}
