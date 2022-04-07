/*Example sketch to control a stepper motor with A4988 stepper motor driver and Arduino without a library. More info: https://www.makerguides.com */

// Define stepper motor connections and steps per revolution:
#define M1dirPin 4
#define M1stepPin 5
#define M2dirPin 8
#define M2stepPin 9

#define enablePin 6
#define stepsPerRevolution 1600

void setup() {
  // Declare pins as output:
  pinMode(M1dirPin, OUTPUT);
  pinMode(M1stepPin, OUTPUT);
  pinMode(M2dirPin, OUTPUT);
  pinMode(M2stepPin, OUTPUT);
  pinMode(enablePin, OUTPUT);
}

void loop() {
  // Set the spinning direction clockwise:
  digitalWrite(M1dirPin, HIGH);
  digitalWrite(M2dirPin, HIGH);
  digitalWrite(enablePin, LOW);

  // Spin the stepper motor 1 revolution slowly:
  for (int i = 0; i < stepsPerRevolution; i++) {
    // These four lines result in 1 step:
    digitalWrite(M1stepPin, HIGH);
    digitalWrite(M2stepPin, HIGH);
    delayMicroseconds(500);
    digitalWrite(M1stepPin, LOW);
    digitalWrite(M2stepPin, LOW);
    delayMicroseconds(500);
  }

  delay(1000);
}
