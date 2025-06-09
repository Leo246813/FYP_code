// Pre-test 2: direction preference for left, right, forward, and backward
// Two modes of actuation are valid for each direction, depends on which one suits the participant

// Define pins for motor driver control
const int motor1_pins[] = {23, 25, 27, 29, 31};  // Pins for motor driver 1
const int motor2_pins[] = {32, 34, 36, 38, 40};  // Pins for motor driver 2
const int motor3_pins[] = {22, 24, 26, 28, 30};  // Pins for motor driver 3

// Define PWM pins
const int pwm_pins[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};  // PWM pins for each motor driver

// Define constants for motor control
const int PWM_MAX = 255;  // Maximum PWM value

// Tap parameters
const int TAP_DURATION = 40;  // Duration of tap movement
const int STOP_DELAY = 40;    // Pause after stop

const int CONST_SPEED = 400;  // Time delay for constant speed
const int SHORT_DELAY = 200;  // Reduced time delay for acceleration
const int MID_DELAY = 500;  // Reduced time delay for acceleration
const int LONG_DELAY = 700;  // Reduced time delay for acceleration

// Constant speed
const int SPEED[] = {CONST_SPEED, CONST_SPEED, CONST_SPEED};
// Acceleration
//const int SPEED[] = {LONG_DELAY, MID_DELAY, SHORT_DELAY};

void setup() {
  // Initialize motor driver pins as outputs
  for (int i = 0; i < sizeof(motor1_pins)/sizeof(motor1_pins[0]); ++i) {
    pinMode(motor1_pins[i], OUTPUT);
    pinMode(motor2_pins[i], OUTPUT);
    pinMode(motor3_pins[i], OUTPUT);   
  }
  
  // Initialize PWM pins as outputs
  for (int i = 0; i < sizeof(pwm_pins)/sizeof(pwm_pins[0]); ++i) {
    pinMode(pwm_pins[i], OUTPUT);
  }
    
  // Standby pins
  digitalWrite(motor1_pins[2], HIGH); // Enable motor driver 1
  digitalWrite(motor2_pins[2], HIGH); // Enable motor driver 2
  digitalWrite(motor3_pins[2], HIGH); // Enable motor driver 3
  
  // RIGHT Direction (2 --> 1 --> 4)
  for (int cycle = 0; cycle < 3; cycle++) {
    tapActuator_twice(motor1_pins[3], motor1_pins[4], pwm_pins[1]); // Actuator 2
    if (cycle == 0) delay(400); else delay(SPEED[0]);
    tapActuator_once(motor1_pins[0], motor1_pins[1], pwm_pins[0]); // Actuator 1
    if (cycle == 0) delay(400); else delay(SPEED[1]);
    tapActuator_once(motor2_pins[3], motor2_pins[4], pwm_pins[9]); // Actuator 4
    if (cycle == 0) delay(400); else delay(SPEED[2]);
    delay(1000);
  }
  delay(500);

  // RIGHT Direction (2 --> 4)
  for (int cycle = 0; cycle < 3; cycle++) {
    tapActuator_twice(motor1_pins[3], motor1_pins[4], pwm_pins[1]); // Actuator 2
    if (cycle == 0) delay(400); else delay(SPEED[0]);
    tapActuator_once(motor2_pins[3], motor2_pins[4], pwm_pins[9]); // Actuator 4
    if (cycle == 0) delay(400); else delay(SPEED[2]);
    delay(1000);
  }
  delay(500);
  
//  // LEFT Direction (4 --> 1 --> 2)
  for (int cycle = 0; cycle < 3; cycle++) {
    tapActuator_twice(motor2_pins[3], motor2_pins[4], pwm_pins[9]); // Actuator 4
    if (cycle == 0) delay(400); else delay(SPEED[0]);
    tapActuator_once(motor1_pins[0], motor1_pins[1], pwm_pins[0]); // Actuator 1
    if (cycle == 0) delay(400); else delay(SPEED[1]);
    tapActuator_once(motor1_pins[3], motor1_pins[4], pwm_pins[1]); // Actuator 2
    if (cycle == 0) delay(400); else delay(SPEED[2]);
    delay(1000);
  }
  delay(500);

  // LEFT Direction (4 --> 2)
  for (int cycle = 0; cycle < 3; cycle++) {
    tapActuator_twice(motor2_pins[3], motor2_pins[4], pwm_pins[9]); // Actuator 4
    if (cycle == 0) delay(400); else delay(SPEED[0]);
    tapActuator_once(motor1_pins[3], motor1_pins[4], pwm_pins[1]); // Actuator 2
    if (cycle == 0) delay(400); else delay(SPEED[2]);
    delay(1000);
  }
  delay(500);

  // BACK Direction (1 --> 2 --> 3)
  for (int cycle = 0; cycle < 3; cycle++) {
    tapActuator_twice(motor1_pins[0], motor1_pins[1], pwm_pins[0]); // Actuator 1
    if (cycle == 0) delay(400); else delay(SPEED[0]);
    tapActuator_once(motor1_pins[3], motor1_pins[4], pwm_pins[1]); // Actuator 2
    if (cycle == 0) delay(400); else delay(SPEED[1]);
    tapActuator_once(motor2_pins[0], motor2_pins[1], pwm_pins[8]); // Actuator 3
    if (cycle == 0) delay(400); else delay(SPEED[2]);
  }
  delay(500);

  // BACK Direction (1 --> 3)
  for (int cycle = 0; cycle < 3; cycle++) {
    tapActuator_twice(motor1_pins[0], motor1_pins[1], pwm_pins[0]); // Actuator 1
    if (cycle == 0) delay(400); else delay(SPEED[0]);
    tapActuator_once(motor2_pins[0], motor2_pins[1], pwm_pins[8]); // Actuator 3
    if (cycle == 0) delay(400); else delay(SPEED[2]);
  }
  delay(500);
  
  // FRONT Direction (3 --> 2 --> 1)
  for (int cycle = 0; cycle < 3; cycle++) {
    tapActuator_twice(motor2_pins[0], motor2_pins[1], pwm_pins[8]); // Actuator 3
    if (cycle == 0) delay(400); else delay(SPEED[0]);
    tapActuator_once(motor1_pins[3], motor1_pins[4], pwm_pins[1]); // Actuator 2
    if (cycle == 0) delay(400); else delay(SPEED[1]);
    tapActuator_once(motor1_pins[0], motor1_pins[1], pwm_pins[0]); // Actuator 1
    if (cycle == 0) delay(400); else delay(SPEED[2]);
  }
  delay(500);

  // FRONT Direction (3 --> 1)
  for (int cycle = 0; cycle < 3; cycle++) {
    tapActuator_twice(motor2_pins[0], motor2_pins[1], pwm_pins[8]); // Actuator 3
    if (cycle == 0) delay(400); else delay(SPEED[0]);
    tapActuator_once(motor1_pins[0], motor1_pins[1], pwm_pins[0]); // Actuator 1
    if (cycle == 0) delay(400); else delay(SPEED[2]);
  }
  delay(500);
}

// === Tap Functions ===
void tapActuator_twice(int IN1, int IN2, int PWM) {
  // Activate
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(PWM, 255);  // 100% PWM
  delay(STOP_DELAY);

  // Stop
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(PWM, 0);
  delay(STOP_DELAY);

  // Activate
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(PWM, 255);  // 100% PWM
  delay(STOP_DELAY);

  // Stop
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(PWM, 0);
  delay(STOP_DELAY);
}

void tapActuator_once(int IN1, int IN2, int PWM) {
  // Activate
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(PWM, 255);  // 100% PWM
  delay(STOP_DELAY);

  // Stop
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(PWM, 0);
  delay(STOP_DELAY);
}

void loop() {
}
