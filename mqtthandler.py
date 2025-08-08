import paho.mqtt.client as mqtt
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
import struct
import json
import logging
from datetime import datetime
import threading
import queue
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class MQTTHandler(QObject):
    data_received = pyqtSignal(str, str, str, int, list, int, int)  # Added frame_index
    connection_status = pyqtSignal(str)

    def __init__(self, db, project_name, broker="192.168.1.231", port=1883):
        super().__init__()
        self.db = db
        self.project_name = project_name
        self.broker = broker
        self.port = port
        self.client = None
        self.connected = False
        self.subscribed_topics = []
        self.data_queue = queue.Queue()
        self.batch_interval_ms = 50
        self.processing_thread = None
        self.running = False
        self.channel_counts = {}
        self._channel_data_buffer = defaultdict(lambda: defaultdict(list))
        self.feature_mapping = {
            "Tabular View": ["TabularView"],
            "Time View": ["TimeWave", "TimeReport"],
            "Time Report": ["TimeReport"],
            "FFT": ["FFT"],
            "Waterfall": ["WaterFall"],
            "Centerline": ["CenterLinePlot"],
            "Orbit": ["OrbitView"],
            "Trend View": ["TrendView"],
            "Multiple Trend View": ["MultiTrendView"],
            "Bode Plot": ["BodePlot"],
            "History Plot": ["HistoryPlot"],
            "Polar Plot": ["PolarPlot"],
            "Report": ["Report"]
        }
        logging.debug(f"Initializing MQTTHandler with project_name: {project_name}, broker: {broker}")

    def parse_topic(self, topic):
        try:
            if not self.db.is_connected():
                self.db.reconnect()
            tag_name = topic
            project_data = self.db.get_project_data(self.project_name)
            if not project_data or "models" not in project_data:
                logging.error(f"No valid project data for {self.project_name}")
                return None, None, None
            model_name = None
            for model in project_data["models"]:
                if model.get("tagName") == topic:
                    model_name = model.get("name")
                    break
            if not model_name:
                logging.warning(f"No model found for topic {topic} in project {self.project_name}")
                return None, None, None
            channel_count_map = {"DAQ4CH": 4, "DAQ8CH": 8, "DAQ10CH": 10}
            raw_channel_count = project_data.get("channel_count", 4)
            try:
                channel_count = channel_count_map.get(raw_channel_count, int(raw_channel_count))
                if channel_count not in [4, 8, 10]:
                    raise ValueError(f"Invalid channel count: {channel_count}")
            except (ValueError, TypeError) as e:
                logging.error(f"Invalid channel count {raw_channel_count}: {str(e)}. Defaulting to 4.")
                channel_count = 4
            self.channel_counts[self.project_name] = channel_count
            logging.debug(f"Parsed topic {topic}: project_name={self.project_name}, model_name={model_name}, tag_name={tag_name}, channels={channel_count}")
            return self.project_name, model_name, tag_name
        except Exception as e:
            logging.error(f"Error parsing topic {topic}: {str(e)}")
            return None, None, None

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            self.connection_status.emit("Connected to MQTT Broker")
            logging.info("Connected to MQTT Broker")
            QTimer.singleShot(0, self.subscribe_to_topics)
        else:
            self.connected = False
            self.connection_status.emit(f"Connection failed with code {rc}")
            logging.error(f"Failed to connect to MQTT Broker with code {rc}")

    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        self.connection_status.emit("Disconnected from MQTT Broker")
        logging.info("Disconnected from MQTT Broker")
        self.subscribed_topics = []

    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = msg.payload
            self.data_queue.put((topic, payload, datetime.now()))
            logging.debug(f"Queued message for topic {topic}, payload size: {len(payload)} bytes")
        except Exception as e:
            logging.error(f"Error queuing MQTT message: {str(e)}")

    def process_data(self):
        batch = defaultdict(list)
        while self.running:
            try:
                start_time = datetime.now()
                while (datetime.now() - start_time).total_seconds() * 1000 < self.batch_interval_ms:
                    try:
                        topic, payload, timestamp = self.data_queue.get(timeout=0.01)
                        batch[topic].append((payload, timestamp))
                    except queue.Empty:
                        continue

                for topic, payloads in batch.items():
                    project_name, model_name, tag_name = self.parse_topic(topic)
                    if not tag_name or project_name != self.project_name or not model_name:
                        logging.warning(f"Skipping invalid topic: {topic}")
                        continue

                    channel_count = self.channel_counts.get(self.project_name, 4)
                    project_data = self.db.get_project_data(self.project_name)
                    model = next((m for m in project_data["models"] if m["name"] == model_name), None)
                    if not model:
                        logging.error(f"Model {model_name} not found")
                        continue
                    expected_channels = len(model.get("channels", []))
                    tacho_channels = 2

                    for payload, _ in payloads:
                        try:
                            values = None
                            sample_rate = 1000
                            frame_index = 0
                            try:
                                payload_str = payload.decode('utf-8')
                                data = json.loads(payload_str)
                                values = data.get("values", [])
                                sample_rate = data.get("sample_rate", 1000)
                                frame_index = data.get("frame_index", 0)
                                if not isinstance(values, list) or len(values) < channel_count:
                                    logging.warning(f"Invalid JSON payload format or insufficient channels: {len(values)}/{channel_count}")
                                    continue
                            except (UnicodeDecodeError, json.JSONDecodeError):
                                payload_length = len(payload)
                                if payload_length < 20 or payload_length % 2 != 0:
                                    logging.warning(f"Invalid payload length: {payload_length} bytes")
                                    continue

                                num_samples = payload_length // 2
                                try:
                                    values = struct.unpack(f"<{num_samples}H", payload)
                                except struct.error as e:
                                    logging.error(f"Failed to unpack payload of {num_samples} uint16_t: {str(e)}")
                                    continue

                                if len(values) < 100:
                                    logging.warning(f"Payload too short: {len(values)} samples")
                                    continue

                                header = values[:100]
                                frame_index = (header[1] << 16) | header[0]  # Combine high and low
                                total_values = values[100:]
                                main_channels = header[2]
                                sample_rate = header[3]
                                tacho_channels_count = header[6]
                                total_channels = main_channels + tacho_channels_count
                                samples_per_channel = (len(total_values) // total_channels) if total_values else 0

                                if main_channels <= 0 or sample_rate <= 0 or tacho_channels_count <= 0 or samples_per_channel <= 0:
                                    logging.error(f"Invalid header: main_channels={main_channels}, sample_rate={sample_rate}, "
                                                 f"tacho_channels_count={tacho_channels_count}, samples_per_channel={samples_per_channel}")
                                    continue

                                if len(total_values) != samples_per_channel * total_channels:
                                    logging.warning(f"Unexpected data length: got {len(total_values)}, expected {samples_per_channel * total_channels}")
                                    continue

                                main_data = total_values[:samples_per_channel * main_channels]
                                tacho_data = total_values[samples_per_channel * main_channels:]
                                channel_data = [[] for _ in range(main_channels)]
                                for i in range(0, len(main_data), main_channels):
                                    for ch in range(main_channels):
                                        channel_data[ch].append(main_data[i + ch])

                                tacho_freq_data = tacho_data[:samples_per_channel] if tacho_channels_count >= 1 else []
                                tacho_trigger_data = tacho_data[samples_per_channel:2 * samples_per_channel] if tacho_channels_count >= 2 else []
                                values = [[float(v) for v in ch] for ch in channel_data]
                                if tacho_freq_data:
                                    values.append([float(v) for v in tacho_freq_data])
                                if tacho_trigger_data:
                                    values.append([float(v) for v in tacho_trigger_data])

                            for feature_name, _ in self.feature_mapping.items():
                                buffer_key = (tag_name, model_name, feature_name)
                                if feature_name == "Multiple Trend View":
                                    if buffer_key not in self._channel_data_buffer:
                                        self._channel_data_buffer[buffer_key] = [[] for _ in range(expected_channels + tacho_channels)]
                                    for ch_idx in range(len(values)):
                                        self._channel_data_buffer[buffer_key][ch_idx].extend(values[ch_idx])
                                    if all(len(ch_data) > 0 for ch_data in self._channel_data_buffer[buffer_key][:-tacho_channels]):
                                        aggregated_values = self._channel_data_buffer[buffer_key]
                                        self.data_received.emit(feature_name, tag_name, model_name, -1, aggregated_values, sample_rate, frame_index)
                                        logging.debug(f"Emitted aggregated data for {feature_name}/{tag_name}/{model_name}: {len(aggregated_values)} channels, frame {frame_index}")
                                        self._channel_data_buffer[buffer_key] = [[] for _ in range(expected_channels + tacho_channels)]
                                else:
                                    if feature_name in ["Time View", "Time Report", "Tabular View"]:
                                        self.data_received.emit(feature_name, tag_name, model_name, -1, values, sample_rate, frame_index)
                                        logging.debug(f"Emitted for {feature_name}/{tag_name}/{model_name}/all_channels: {len(values)} channels, frame {frame_index}")
                                    elif feature_name in ["Orbit", "FFT"]:
                                        for ch_idx in range(min(channel_count, len(values))):
                                            channel_values = values[ch_idx] if ch_idx < len(values) else []
                                            self.data_received.emit(feature_name, tag_name, model_name, ch_idx, channel_values, sample_rate, frame_index)
                                            logging.debug(f"Emitted for {feature_name}/{tag_name}/{model_name}/channel_{ch_idx}: {len(channel_values)} samples, frame {frame_index}")
                                    else:
                                        for ch_idx in range(min(channel_count, len(values))):
                                            channel_values = values[ch_idx] if ch_idx < len(values) else []
                                            self.data_received.emit(feature_name, tag_name, model_name, ch_idx, channel_values, sample_rate, frame_index)
                                            logging.debug(f"Emitted for {feature_name}/{tag_name}/{model_name}/channel_{ch_idx}: {len(channel_values)} samples, frame {frame_index}")

                        except Exception as e:
                            logging.error(f"Error processing payload for topic {topic}: {str(e)}")

                batch.clear()
            except Exception as e:
                logging.error(f"Error in data processing loop: {str(e)}")
                self.connection_status.emit(f"Data processing error: {str(e)}")

    def subscribe_to_topics(self):
        try:
            if not self.db.is_connected():
                self.db.reconnect()
            project_data = self.db.get_project_data(self.project_name)
            for model in project_data.get("models", []):
                tag_name = model.get("tagName", "")
                if tag_name and tag_name not in self.subscribed_topics:
                    self.client.subscribe(tag_name)
                    self.subscribed_topics.append(tag_name)
                    logging.info(f"Subscribed to topic: {tag_name}")
        except Exception as e:
            logging.error(f"Error subscribing to topics: {str(e)}")
            self.connection_status.emit(f"Failed to subscribe to topics: {str(e)}")

    def start(self):
        try:
            self.client = mqtt.Client()
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_message = self.on_message
            self.client.connect_async(self.broker, self.port, 60)
            self.client.loop_start()
            self.running = True
            self.processing_thread = threading.Thread(target=self.process_data, daemon=True)
            self.processing_thread.start()
            logging.info("MQTT client and processing thread started")
        except Exception as e:
            logging.error(f"Failed to start MQTT client: {str(e)}")
            self.connection_status.emit(f"Failed to start MQTT: {str(e)}")

    def stop(self):
        try:
            self.running = False
            if self.processing_thread:
                self.processing_thread.join(timeout=1.0)
                self.processing_thread = None
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
                self.connected = False
                self.subscribed_topics = []
                logging.info("MQTT client and processing thread stopped")
        except Exception as e:
            logging.error(f"Error stopping MQTT client: {str(e)}")