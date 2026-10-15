int ledPin = 13;  // Built-in LED on pin 13

void setup() {
  pinMode(ledPin, OUTPUT); // Set LED pin as output
  Serial.begin(9600);      // Start serial communication at 9600 bps
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); // Read the incoming command

    if (command == "turn on the led") {
      digitalWrite(ledPin, HIGH);  // Turn LED ON
    }
    else if (command == "turn off the led") {
      digitalWrite(ledPin, LOW);   // Turn LED OFF
    }
  }
}
