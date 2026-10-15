import serial
import speech_recognition as sr
import pyttsx3
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to convert text to speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to take voice commands
def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")
    except Exception as e:
        print("Sorry, I didn't get that. Please repeat.")
        return "None"
    
    return query.lower()

# Setup Serial Communication with Arduino
arduino = serial.Serial('COM3', 9600, timeout=1)  # Replace 'COM3' with your Arduino's port
time.sleep(2)  # Wait for the serial connection to initialize

# Main function
if __name__ == "__main__":
    speak("I am ready to control the LED.")
    while True:
        command = take_command()

        # Voice commands to control the LED
        if "turn on the led" in command:
            arduino.write(b'turn on the led\n')  # Send command to Arduino
            speak("Turning on the LED.")
        elif "turn off the led" in command:
            arduino.write(b'turn off the led\n')  # Send command to Arduino
            speak("Turning off the LED.")
        elif "stop" in command or "exit" in command:
            speak("Goodbye!")
            break
        else:
            speak("Sorry, I didn't understand that command.")
