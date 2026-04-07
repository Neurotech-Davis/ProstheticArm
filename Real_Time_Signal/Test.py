import serial
import time

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/cu.SLAB_USBtoUART' # Make sure to use your actual Mac port here!
BAUD_RATE = 115200

def main():
    try:
        print(f"Connecting to {SERIAL_PORT}...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Wait for the board's auto-reset to finish
        print("Connected! Ready to send LDA triggers.")

        while True:
            # We only have one command now: sending the trigger.
            command = input("Enter '1' to simulate LDA 'eyes closed' detection (or 'q' to quit): ")

            if command == 'q':
                print("Exiting...")
                break
            
            elif command == '1':
                # Send the single trigger signal
                ser.write(b'1')
                print("Trigger sent to board.")
                
                # Wait a tiny fraction of a second for the board to process and reply
                time.sleep(0.05)
                
                # Read back the new state from the board
                if ser.in_waiting > 0:
                    response = ser.readline().decode('utf-8').strip()
                    print(f"Board response: {response}")
            else:
                print("Invalid input. Just press '1' to trigger or 'q' to quit.")

    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial connection closed. (Board will likely auto-reset to OPEN state)")

if __name__ == '__main__':
    main()