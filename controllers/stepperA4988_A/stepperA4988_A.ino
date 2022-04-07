// Include the Stepper library:
#include <AccelStepper.h>

// Give the motors control pins names:
#define M1dirPin 4
#define M1stepPin 5

#define M2dirPin 8
#define M2stepPin 9

// Define number of steps per revolution:
int defaultSpeed = 20;
int ind = 0;

int dir = 1;   // rotation direction 1 or -1
int steps = 0; // number of steps to rotate
int spd = 20;  // speed in steps per second
long curPos = 0;  // curent position of the motor
long newPos = 0;  // curent position of the motor

String command; // input command

// Initialize the stepper library on the motor shield:
AccelStepper stepper1 = AccelStepper(1, M1stepPin, M1dirPin);
AccelStepper stepper2 = AccelStepper(1, M2stepPin, M2dirPin);

void setup() {
  pinMode(6, OUTPUT); // Enable - if connected
  pinMode(10, OUTPUT); // Enable - if connected
  
  // Set the motor max and default speed (RPMs):
  stepper1.setMaxSpeed(600);
  stepper1.setSpeed(defaultSpeed);
  stepper1.setCurrentPosition(0);

  stepper2.setMaxSpeed(600);
  stepper2.setSpeed(defaultSpeed);
  stepper2.setCurrentPosition(0);

  Serial.begin(9600);
}

/*
 * Rotate 2 motors simultaneously
 */
void rotate(int dir, int steps, int spd) {
    curPos = stepper1.currentPosition();  // should be same for both motors
    newPos = curPos + (dir * steps);
    
    stepper1.setSpeed(dir * spd);
    stepper2.setSpeed(dir * spd);

    while ((stepper1.currentPosition() != newPos) || (stepper2.currentPosition() != newPos)) {
      digitalWrite(6, LOW); // Set Enable low
      digitalWrite(10, LOW); // Set Enable low
        
      if (stepper1.currentPosition() != newPos) {
        stepper1.runSpeed();
      }
      
      if (stepper2.currentPosition() != newPos) {
        stepper2.runSpeed();
      }
    }
}


void loop() {
  digitalWrite(6, HIGH); // turn off motor drivers
  digitalWrite(10, HIGH); // turn off motor drivers
        
  if (Serial.available())  {
    char c = Serial.read();  //gets one byte from serial buffer
    
    if (c == '\n') {
      //Serial.println(command); //prints string to serial port out
      
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
      Serial.println(stepper1.currentPosition());
    }  
    else {     
      command += c; //makes the string readString
    }
  }
}
