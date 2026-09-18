"""FastF1 telemetry preparation and SocketCAN replay tools."""

SEQUENCE_LENGTH = 10

WINDOW_SIZE = 20  # 20 timesteps at 10 Hz = 2.0 seconds of history
STRIDE = 1        # Step size for sliding window
BATCH_SIZE = 64

PROCESSED_COLUMNS = ["DeltaTime"]#processed from data in fastf1, so telemetry changes appear here
#this processing is done in telemetry.py