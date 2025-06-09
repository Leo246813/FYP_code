/*
 * Tactile Pattern Controller
 * 
 * This Arduino sketch controls 7 actuators (vibration motors) for a tactile pattern recognition study.
 * It can receive pattern commands via Serial from a web interface
 * or run built-in patterns for testing purposes.
 */

// Pin definitions for the 7 actuators
const int motor1_pins[] = {23, 25, 27, 29, 31};  // Pins for motor driver 1
const int motor2_pins[] = {32, 34, 36, 38, 40};  // Pins for motor driver 2
const int motor3_pins[] = {22, 24, 26, 28, 30};  // Pins for motor driver 3
const int motor4_pins[] = {42, 44, 46, 48, 50};  // Pins for motor driver 4

const int in1Pins[] = {23, 29, 32, 38, 22, 28, 42};
const int in2Pins[] = {25, 31, 34, 40, 24, 30, 44};

const int pwm_pins[] = {2, 3, 10, 11, 6, 7, 8};  // PWM pins for actuators 1-7
const int numActuators = 7; // Last actuator is for danger detection only

// Parameters for pattern execution
const int TAP_DURATION = 40;          // Duration of tap movement
const int STOP_DELAY = 40;            // Pause after stop
const int patternDuration = 500;      // Duration of each actuator in a pattern (ms)
const int patternPause = 400;         // Pause between activations in a sequence (ms)
const int maximumStrength = 255;    // PWM value for vibration intensity (0-255)
const int mediumStrength = 192;       // Medium PWM strength (75%)
//const int vibrationStrength = mediumStrength;
const int vibrationStrength = maximumStrength;

// Buffer for receiving serial commands
const int bufferSize = 32;
char inputBuffer[bufferSize];
int bufferIndex = 0;

// For tracking experiment patterns
int sequenceCount = 0;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  
  // Initialize motor driver pins as outputs
  for (int i = 0; i < sizeof(motor1_pins)/sizeof(motor1_pins[0]); ++i) {
    pinMode(motor1_pins[i], OUTPUT);
    pinMode(motor2_pins[i], OUTPUT);
    pinMode(motor3_pins[i], OUTPUT);   
    pinMode(motor4_pins[i], OUTPUT);
  }

  // Initialize PWM pins as outputs
  for (int i = 0; i < numActuators; i++) {
    pinMode(pwm_pins[i], OUTPUT);
  }

  // Standby pins - enable all motor drivers
  digitalWrite(motor1_pins[2], HIGH); // Enable motor driver 1
  digitalWrite(motor2_pins[2], HIGH); // Enable motor driver 2
  digitalWrite(motor3_pins[2], HIGH); // Enable motor driver 3
  digitalWrite(motor4_pins[2], HIGH); // Enable motor driver 4
  
  // Send ready message to the web interface
  Serial.println("Arduino ready");
  
  // Uncomment to run a test pattern at startup
  // runTestPattern();
}

// Activate a single actuator with tapping behavior (once)
void tapActuator_once(int actuatorIndex, int strength = mediumStrength) {
  // Check if the actuator index is valid (0-5)
  if (actuatorIndex >= 0 && actuatorIndex < numActuators) {
    int pwmPin = pwm_pins[actuatorIndex];
    int in1Pin = in1Pins[actuatorIndex];
    int in2Pin = in2Pins[actuatorIndex];
    
    // Activate
    digitalWrite(in1Pin, HIGH);
    digitalWrite(in2Pin, LOW);
    analogWrite(pwmPin, strength);
    delay(TAP_DURATION);
    
    // Stop
    digitalWrite(in1Pin, LOW);
    digitalWrite(in2Pin, LOW);
    analogWrite(pwmPin, 0);
  }
}

// Activate a single actuator with tapping behavior (twice)
void tapActuator_twice(int actuatorIndex, int strength = mediumStrength) {
  // Check if the actuator index is valid (0-5)
  if (actuatorIndex >= 0 && actuatorIndex < numActuators) {
    int pwmPin = pwm_pins[actuatorIndex];
    int in1Pin = in1Pins[actuatorIndex];
    int in2Pin = in2Pins[actuatorIndex];
    
    // First tap
    digitalWrite(in1Pin, HIGH);
    digitalWrite(in2Pin, LOW);
    analogWrite(pwmPin, strength);
    delay(TAP_DURATION);
    
    // Stop
    digitalWrite(in1Pin, LOW);
    digitalWrite(in2Pin, LOW);
    analogWrite(pwmPin, 0);
    delay(STOP_DELAY);

    // Second tap
    digitalWrite(in1Pin, HIGH);
    digitalWrite(in2Pin, LOW);
    analogWrite(pwmPin, strength);
    delay(TAP_DURATION);
    
    // Stop
    digitalWrite(in1Pin, LOW);
    digitalWrite(in2Pin, LOW);
    analogWrite(pwmPin, 0);
  }
}


// Pattern 1: Constant Speed - Sequential activation for 2 cycles
void runConstantSpeed(char* command) {
  // Default values
  int vibrationStrength = mediumStrength;  // choose from: maximumStrength/mediumStrength
  
  // Parse the command (format: "P123" or "CS123")
  int selectedActuators[numActuators];
  int numSelected = 0;
  
  // Find the start of the actuator numbers
  char* actuatorStr = command;
  while (*actuatorStr != '\0' && !isdigit(*actuatorStr)) {
    actuatorStr++;
  }
  
  // Parse the actuator numbers
  while (*actuatorStr != '\0' && numSelected < numActuators) {
    int val = *actuatorStr - '0';  // Convert char digit to int
    if (val >= 1 && val <= numActuators) {
      selectedActuators[numSelected++] = val - 1;  // Convert to 0-based index
    }
    actuatorStr++;
  }
  
  // If no actuators specified, use a default set of 3
  if (numSelected == 0) {
    selectedActuators[0] = 0;  // Actuator 1
    selectedActuators[1] = 1;  // Actuator 2
    selectedActuators[2] = 2;  // Actuator 3
    numSelected = 3;
  }
  
  Serial.println("Running Pattern: Constant Speed");
  Serial.print("Selected actuators: ");
  for (int i = 0; i < numSelected; i++) {
    Serial.print(selectedActuators[i] + 1);
    if (i < numSelected - 1) Serial.print(", ");
  }
  Serial.println();
  
  // Execute pattern for exactly 3 cycles
  for (int cycle = 1; cycle <= 3; cycle++) {
    Serial.print("Cycle ");
    Serial.println(cycle);
    
    // First actuator taps twice (to show direction)
    tapActuator_twice(selectedActuators[0], vibrationStrength);
    delay(patternPause);
    
    // Remaining actuators tap once
    for (int i = 1; i < numSelected; i++) {
      tapActuator_once(selectedActuators[i], vibrationStrength);
      delay(patternPause);
    }
    
    delay(1000); // Inter-cycle pause
  }

  
  Serial.println("Pattern complete");
}

// Pattern 2: Acceleration/Deceleration - Varying time between activations for 2 cycles
void runAccelerationDeceleration(char* params) {
  // Parse the parameters
  bool isAcceleration = true; // Default to acceleration
  
  // Parse actuator selection (comma-separated list of actuator indices starting from 1)
  int selectedActuators[numActuators];
  int numSelected = 0;
  
  char* token = strtok(params, ",");
  while (token != NULL && numSelected < numActuators) {
    // Check if token is "acc" or "dec" for acceleration/deceleration
    if (strcmp(token, "dec") == 0) {
      isAcceleration = false;
    }
    else if (strcmp(token, "acc") == 0) {
      isAcceleration = true;
    }
    else {
      int val = atoi(token);
      // It's an actuator number (1-7)
      if (val >= 1 && val <= numActuators) {
        selectedActuators[numSelected++] = val - 1; // Convert to 0-based index
      }
    }
    token = strtok(NULL, ",");
  }
  
  // If no actuators specified, use a default set of 3
  if (numSelected == 0) {
    selectedActuators[0] = 0; // Actuator 1
    selectedActuators[1] = 1; // Actuator 2
    selectedActuators[2] = 2; // Actuator 3
    numSelected = 3;
  }
  
  Serial.println("Running Pattern: Acceleration/Deceleration");
  Serial.print(isAcceleration ? "Accelerating" : "Decelerating");
  Serial.println(" pattern");
  
  Serial.print("Selected actuators: ");
  for (int i = 0; i < numSelected; i++) {
    Serial.print(selectedActuators[i] + 1);
    if (i < numSelected - 1) Serial.print(", ");
  }
  Serial.println();
  
  // Define delays based on number of actuators
  int delays[4]; // Support up to 4 actuators
  
  if (numSelected == 2) {
    if (isAcceleration) {
      delays[0] = 400; // slow
      delays[1] = 200; // fast
    } else {
      delays[0] = 500; // fast
      delays[1] = 1000; // slow
    }
  } else if (numSelected == 3) {
    if (isAcceleration) {
      delays[0] = 600; // slow
      delays[1] = 400; // medium
      delays[2] = 200; // fast
    } else {
      delays[0] = 500; // fast
      delays[1] = 800; // medium
      delays[2] = 1200; // slow
    }
  } else if (numSelected == 4) {
    if (isAcceleration) {
      delays[0] = 800; // slowest
      delays[1] = 600; // slow
      delays[2] = 400; // medium
      delays[3] = 200; // fast
    } else {
      delays[0] = 500; // fast
      delays[1] = 800; // medium
      delays[2] = 1200; // slow
      delays[3] = 1500; // slowest
    }
  } else {
    // Default case for other numbers of actuators
    for (int i = 0; i < numSelected; i++) {
      delays[i] = 800; // constant medium delay
    }
  }
  
  // Execute pattern for exactly 3 cycles
  for (int cycle = 1; cycle <= 3; cycle++) {
    Serial.print("Cycle ");
    Serial.println(cycle);
    
    for (int i = 0; i < numSelected; i++) {
      // First actuator in each cycle gets double tap
      if (i == 0) {
        tapActuator_twice(selectedActuators[i]);
      } else {
        tapActuator_once(selectedActuators[i]);
      }
      
      // Apply delay based on cycle
      if (!(cycle == 3 && i == numSelected - 1)) { // Don't delay after last actuator of last cycle
        if (cycle == 1) {
          // First cycle: constant speed (minimal delay)
          delay(800);
        } else {
          // Second and third cycles: use acceleration/deceleration delays
          delay(delays[i]);
        }
      }
    }
    
    // Pause between cycles (except after last cycle)
    if (cycle < 4) {
      delay(1000); // Inter-cycle pause
    }
  }
  
  Serial.println("Pattern complete");
}

// Pattern 3: Danger Detection - Warning actuator activates 3 times, danger actuator for 3 seconds
void runDangerDetection(char* params) {
  // Parse parameters
  int dangerActuator = numActuators - 1; // Selected actuator (from 1-6)
  int warningActuator = 6; // Default to last actuator
  
  // Parse comma-separated parameters
  char* token = strtok(params, ",");
  int paramCount = 0;
  
  while (token != NULL && paramCount < 2) {
    int val = atoi(token);
    
    if (paramCount == 0) {
      // First param is danger actuator
      if (val >= 1 && val <= numActuators) {
        dangerActuator = val - 1; // Convert to 0-based
      }
    } else if (paramCount == 1) {
      // Second param is warning actuator
      if (val >= 1 && val <= numActuators) {
        warningActuator = val - 1; // Convert to 0-based
      }
    }
    
    token = strtok(NULL, ",");
    paramCount++;
  }
  
  Serial.println("Running Pattern: Danger Detection");
  Serial.print("Danger actuator: ");
  Serial.print(dangerActuator + 1);
  Serial.print(", Warning actuator: ");
  Serial.println(warningActuator + 1);
  
  // Both warning and danger actuators activate simultaneously for 3 seconds
  Serial.println("Warning + Danger phase: simultaneous activation for 3 seconds");
  Serial.println("Warning: 3 double taps (1 per second), Danger: continuous at 5Hz");
  
  int startTime = millis();
  int currentTime;
  int lastDangerTap = startTime;
  int warningCount = 0;
  int nextWarningTime = startTime + 1000; // First warning at 1 second
      
  while ((currentTime = millis()) - startTime < 3000) { // 3 seconds total
    // Danger actuator every 200ms (5Hz frequency)
    if (currentTime - lastDangerTap >= 100) {
      tapActuator_once(dangerActuator, maximumStrength);
      lastDangerTap = currentTime;
    }
    
    // Warning actuator 3 times, evenly spaced (at 1s, 2s, 3s)
    if (warningCount < 3 && currentTime >= nextWarningTime) {
      tapActuator_once(warningActuator, vibrationStrength);
      warningCount++;
      nextWarningTime = startTime + (warningCount + 1) * 1000; // Next warning in 1 second
    }
  }
  
  Serial.println("Pattern complete");
}


// Process incoming commands from the web interface
void processCommand(char* command) {
  Serial.print("Received command: ");
  Serial.println(command);
  
  // Check command type based on prefix
//  if (strncmp(command, "P:", 2) == 0) {
//    // P: Pattern command - execute a sequence of actuators
//    executePattern(&command[2]);  // Pass everything after "P:"
//  }
  if (strncmp(command, "CS:", 3) == 0) {
    // CS: Constant Speed pattern
    runConstantSpeed(&command[3]);
  }
  else if (strncmp(command, "AD:", 3) == 0) {
    // AD: Acceleration/Deceleration pattern
    runAccelerationDeceleration(&command[3]);
  }
  else if (strncmp(command, "DD:", 3) == 0) {
    // DD: Danger Detection
    runDangerDetection(&command[3]);
  }
  else {
    Serial.println("Unknown command");
  }
}

void loop() {
  // Check for incoming serial data
  if (Serial.available() > 0) {
    char c = Serial.read();
    
    // Process character
    if (c == '\n') {
      // End of command, process it
      inputBuffer[bufferIndex] = '\0';  // Null-terminate the string
      processCommand(inputBuffer);
      bufferIndex = 0;  // Reset buffer for next command
    } 
    else if (bufferIndex < bufferSize - 1) {
      // Add character to buffer
      inputBuffer[bufferIndex++] = c;
    }
  }
}
