/*
 * Serial interface:
 * - activate / deactivate IR LEDs: 'd'
 * - send command to both motors: '+200:50', where
 *      - '+' or '-' is the direction (mandatory), - CW, + CCW
 *      - '200' is steps to move
 *      - '50' is a speed
 *   which reads as "Move 200 steps at a speed of 50 counterclockwise"
 */


// Include the Stepper library:
#include <AccelStepper.h>

// Give the motors control pins names:
#define M1dirPin 4
#define M1stepPin 5

#define M2dirPin 8
#define M2stepPin 9

#define M1diodePin 11
#define M2diodePin 13

// Define number of steps per revolution:
int defaultSpeed = 20;
int ind = 0;

int dir = 1;   // rotation direction 1 or -1
int steps = 0; // number of steps to rotate
int spd = 20;  // speed in steps per second
long curPos = 0;  // curent position of the motor
long newPos = 0;  // curent position of the motor
unsigned long currTime = 0; // tracking time since the beginning for blinking
int diodesOn = 0;  // diodes on / off
int blinkInterval = 3000;  // in millis
int blinkDuration = 500;   // in millis

String command; // input command

// Initialize the stepper library on the motor shield:
AccelStepper stepper1 = AccelStepper(1, M1stepPin, M1dirPin);
AccelStepper stepper2 = AccelStepper(1, M2stepPin, M2dirPin);

void setup() {
  pinMode(6, OUTPUT); // Enable - if connected
  pinMode(10, OUTPUT);
  pinMode(M1diodePin, OUTPUT);
  pinMode(M2diodePin, OUTPUT);

  updateLEDs();
  
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
      updateLEDs();
      
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

/*
 * Blink the motor LEDs
 */
void updateLEDs() {
  currTime = millis();

  if ((currTime % blinkInterval < blinkDuration) && (diodesOn == 1))  {
    digitalWrite(M1diodePin, HIGH); // Set diodes to HIGH
    digitalWrite(M2diodePin, HIGH); // Set diodes to HIGH
  } else {
    digitalWrite(M1diodePin, LOW); // Set diodes to LOW
    digitalWrite(M2diodePin, LOW); // Set diodes to LOW    
  }
}


void loop() {
  updateLEDs();
  
  digitalWrite(6, HIGH); // turn off motor drivers
  digitalWrite(10, HIGH); // turn off motor drivers
        
  if (Serial.available())  {
    char c = Serial.read();  // gets one byte from serial buffer
    
    if (c == '\n') {
      //Serial.println(command); // prints string to serial port out

      String dirCmd = command.substring(0, 1);

      // turn LEDs on
      if (dirCmd.equals("d")) {
        diodesOn = (diodesOn == 1) ? 0 : 1;

      // rotate motors
      } else {
        
        // rotation direction
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
      }
      command = ""; //clears variable for new input

      // output of the current position
      Serial.println(stepper1.currentPosition());
    }  
    else {     
      command += c; //makes the string readString
    }
  }
}
