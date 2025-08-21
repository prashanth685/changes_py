import math
import struct
import paho.mqtt.publish as publish
from PyQt5.QtCore import QTimer, QObject
from PyQt5.QtWidgets import QApplication
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class MQTTPublisher(QObject):
    def __init__(self, broker, topics):
        super().__init__()
        self.broker = broker
        self.topics = topics if isinstance(topics, list) else [topics]
        self.count = 1

        self.frequency =  500 # Hz
        self.amplitude = 1.5  # Reduced amplitude for clarity (0-3.3V range)
        self.amplitude_scaled = (self.amplitude*0.5) / (3.3 / 65535)  # Scale for 16-bit ADC
        self.offset = 32768  # Midpoint for 16-bit unsigned (0-65535)
        self.sample_rate = 4096  # Samples per second
        self.time_per_message = 1.0  # 1 second for 4096 samples
        self.current_time = 0.0
        self.num_channels = 4 # Main channels
        self.samples_per_channel = 4096  # Samples per channel
        self.num_tacho_channels = 2  # Tacho freq + tacho trigger
        self.frame_index = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.publish_message)
        self.timer.start(1000)  # Publish every 1 second
        logging.debug(f"Initialized MQTTPublisher with broker: {self.broker}, topics: {self.topics}")

    def publish_message(self):
        try:
            # Generate sine wave samples for all main channels
            all_channel_data = []
            for i in range(self.samples_per_channel):
                t = self.current_time + (i / self.sample_rate)
                base_value = self.offset + self.amplitude_scaled * math.sin(2 * math.pi * self.frequency * t)
                rounded_value = int(round(base_value))
                all_channel_data.append(rounded_value)

            self.current_time += self.time_per_message

            # Interleave channel data (16384 = 4096 samples * 4 channels)
            interleaved = []
            for i in range(self.samples_per_channel):
                for ch in range(self.num_channels):
                    interleaved.append(all_channel_data[i])  # Same data for all channels

            if len(interleaved) != self.samples_per_channel * self.num_channels:
                logging.error(f"Interleaved data length incorrect: expected {self.samples_per_channel * self.num_channels}, got {len(interleaved)}")
                return

            # Generate tacho frequency data (4096 samples, constant frequency)
            tacho_freq_data = [self.frequency] * self.samples_per_channel

            # Generate tacho trigger data (10 pulses at 1.0)
            tacho_trigger_data = [0] * self.samples_per_channel
            num_triggers = self.frequency  # 10 triggers per second
            step = self.samples_per_channel // num_triggers  # ~409 samples apart
            for i in range(num_triggers):
                index = i * step
                if index < self.samples_per_channel:
                    tacho_trigger_data[index] = 1  # Pulse at 1.0

            # Build header
            header = [
                self.frame_index % 65535,  # Frame index low
                self.frame_index // 65535,  # Frame index high
                self.num_channels,         # Number of channels (4)header[2]
                self.sample_rate,          # Sample rate (4096)
                4096,                        # Bit depth
                self.samples_per_channel,  # Samples per channel (4096)
                self.num_tacho_channels,   # Number of tacho channels (2)header[6]
                0, 0, 0                   # Reserved
            ]
            while len(header) < 100:
                header.append(0)

            # Combine all data
            message_values = header + interleaved + tacho_freq_data + tacho_trigger_data
            total_expected = 100 + (self.samples_per_channel * self.num_channels) + (self.samples_per_channel * self.num_tacho_channels)
            if len(message_values) != total_expected:
                logging.error(f"Message length incorrect: expected {total_expected}, got {len(message_values)}")
                return

            # Log sample data for debugging
            logging.debug(f"Header: {header}")
            logging.debug(f"Main channel data (first 5): {interleaved[:5]}")
            logging.debug(f"Tacho freq data (first 5): {tacho_freq_data[:5]}")
            logging.debug(f"Tacho trigger data (first 20): {tacho_trigger_data[:20]}")

            # Convert to binary
            binary_message = struct.pack(f"<{len(message_values)}H", *message_values)

            # Publish to all topics
            for topic in self.topics:
                try:
                    publish.single(topic, binary_message, hostname=self.broker, qos=1)
                    logging.info(f"[{self.count}] Published to {topic}: frame {self.frame_index}, {len(message_values)} values")
                except Exception as e:
                    logging.error(f"Failed to publish to {topic}: {str(e)}")

            self.frame_index += 1
            self.count += 1
        except Exception as e:
            logging.error(f"Error in publish_message: {str(e)}")

if __name__ == "__main__":
    app = QApplication([])
    broker = "192.168.1.231"
    topics = ["sarayu/d1/topic1"]
    mqtt_publisher = MQTTPublisher(broker, topics)
    app.exec_()