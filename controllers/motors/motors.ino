// Include the Stepper library:
#include <Stepper.h>

// Define number of steps per revolution:
const int stepsPerRevolution = 20;
int defaultSpeed = 20;
int ind = 0;

int dir = 1;   // rotation direction 1 or -1
int steps = 0; // number of steps to rotate
int spd = 20;  // speed in steps per second


String command; // input command

// Give the motor control pins names:
#define pwmA 3
#define pwmB 11
#define brakeA 9
#define brakeB 8
#define dirA 5
#define dirB 4

// Initialize the stepper library on the motor shield:
Stepper myStepper = Stepper(stepsPerRevolution, dirA, dirB);

void setup() {
  // Set the PWM and brake pins so that the direction pins can be used to control the motor:
  pinMode(pwmA, OUTPUT);
  pinMode(pwmB, OUTPUT);
  pinMode(brakeA, OUTPUT);
  pinMode(brakeB, OUTPUT);
  digitalWrite(pwmA, HIGH);
  digitalWrite(pwmB, HIGH);
  digitalWrite(brakeA, LOW);
  digitalWrite(brakeB, LOW);
  
  // Set the motor defaul speed (RPMs):
  myStepper.setSpeed(defaultSpeed);

  Serial.begin(9600);
}


void rotate(int dir, int steps, int spd) {
      myStepper.setSpeed(spd);
      myStepper.step(dir * steps);
      myStepper.setSpeed(defaultSpeed);
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
    }  
    else {     
      command += c; //makes the string readString
    }
  }
}
