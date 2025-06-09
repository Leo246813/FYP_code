// Define pins for motor driver control
const int motor1_pins[] = {23, 25, 27, 29, 31};  // Pins for motor driver 1
const int motor2_pins[] = {32, 34, 36, 38, 40};  // Pins for motor driver 2
const int motor3_pins[] = {22, 24, 26, 28, 30};  // Pins for motor driver 3
const int motor4_pins[] = {42, 44, 46, 48, 50};  // Pins for motor driver 4

// Define PWM pins
const int pwm_pins[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};  // PWM pins for each motor driver

// Define constants for motor control
// Maximum PWM value: 255; 75% PWM: 192; 50% PWM: 128; 25% PWM: 64
const int PWMValue = 255; 

// Tap parameters
const int TAP_DURATION = 40;  // Duration of tap movement
const int STOP_DELAY = 40;    // Pause after stop

void setup() {
  // Initialize motor driver pins as outputs
  for (int i = 0; i < sizeof(motor1_pins)/sizeof(motor1_pins[0]); ++i) {
    pinMode(motor1_pins[i], OUTPUT);
    pinMode(motor2_pins[i], OUTPUT);
    pinMode(motor3_pins[i], OUTPUT);   
    pinMode(motor4_pins[i], OUTPUT);
  }
  
  // Initialize PWM pins as outputs
  for (int i = 0; i < sizeof(pwm_pins)/sizeof(pwm_pins[0]); ++i) {
    pinMode(pwm_pins[i], OUTPUT);
  }
    
  // Standby pins
  digitalWrite(motor1_pins[2], HIGH); // Enable motor driver 1
  digitalWrite(motor2_pins[2], HIGH); // Enable motor driver 2
  digitalWrite(motor3_pins[2], HIGH); // Enable motor driver 3
  digitalWrite(motor4_pins[2], HIGH); // Enable motor driver 4

  // Equal force testing with single tap & max amplitude
//  singleTap(motor1_pins[0], motor1_pins[1], pwm_pins[0]); // act 1
//  delay(1000);
//  singleTap(motor1_pins[3], motor1_pins[4], pwm_pins[1]); // act 2
//  delay(1000);
//  singleTap(motor2_pins[0], motor2_pins[1], pwm_pins[8]); // act 3
//  delay(1000);
//  singleTap(motor2_pins[3], motor2_pins[4], pwm_pins[9]); // act 4
//  delay(1000);
//  singleTap(motor3_pins[0], motor3_pins[1], pwm_pins[4]); // act 5
//  delay(1000);
//  singleTap(motor3_pins[3], motor3_pins[4], pwm_pins[5]); // act 6
//  delay(1000);


  // Number of taps (single tap, double taps)
  singleTap(motor1_pins[0], motor1_pins[1], pwm_pins[0]); // act 1
  delay(1000);
  doubleTap(motor1_pins[0], motor1_pins[1], pwm_pins[0]); // act 1
  delay(1000);
  singleTap(motor1_pins[3], motor1_pins[4], pwm_pins[1]); // act 2
  delay(1000);
  doubleTap(motor1_pins[3], motor1_pins[4], pwm_pins[1]); // act 2
  delay(1000);
  singleTap(motor2_pins[0], motor2_pins[1], pwm_pins[8]); // act 3
  delay(1000);
  doubleTap(motor2_pins[0], motor2_pins[1], pwm_pins[8]); // act 3
  delay(1000);
  singleTap(motor2_pins[3], motor2_pins[4], pwm_pins[9]); // act 4
  delay(1000);
  doubleTap(motor2_pins[3], motor2_pins[4], pwm_pins[9]); // act 4
  delay(1000);
  singleTap(motor3_pins[0], motor3_pins[1], pwm_pins[4]); // act 5
  delay(1000);
  doubleTap(motor3_pins[0], motor3_pins[1], pwm_pins[4]); // act 5
  delay(1000);
  singleTap(motor3_pins[3], motor3_pins[4], pwm_pins[5]); // act 6
  delay(1000);
  doubleTap(motor3_pins[3], motor3_pins[4], pwm_pins[5]); // act 6
  delay(1000);
}

// === Tap Functions ===

void tapActuator(int IN1, int IN2, int PWM) {
  // Activate (choose one direction)
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(PWM, PWMValue);
  delay(TAP_DURATION);

  // Stop
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(PWM, 0);
  delay(STOP_DELAY);
  
  // Stop the actuator
  digitalWrite(IN2, LOW);
  analogWrite(PWM, 0);
  delay(40);
}

void tapActuator_double(int IN1, int IN2, int PWM) {
  // Activate (choose one direction)
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(PWM, PWMValue);
  delay(STOP_DELAY);

  // Stop
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(PWM, 0);
  delay(STOP_DELAY);

  // DOUBLE TAP (comment out if necessary)
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  analogWrite(PWM, PWMValue);
  delay(STOP_DELAY);
  
  // Stop the actuator
  digitalWrite(IN2, LOW);
  analogWrite(PWM, 0);
  delay(STOP_DELAY);
}

void singleTap(int IN1, int IN2, int PWM) {
  tapActuator(IN1, IN2, PWM);
}

void doubleTap(int IN1, int IN2, int PWM) {
  tapActuator_double(IN1, IN2, PWM);
}

void loop() {
}
