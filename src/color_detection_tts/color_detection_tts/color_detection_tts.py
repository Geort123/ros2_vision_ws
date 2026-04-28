#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
venv_path = '/home/admin/ros2_vision_ws/jt/lib/python3.12/site-packages'
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.insert(0, venv_path)
import cv2
import numpy as np
import threading
import pyttsx3
import time

class SimpleColorTTS(Node):
    def __init__(self):
        super().__init__('simple_color_tts')
        
        # ========== SIMPLE THREADING ==========
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.processing = True
        
        # ========== VISION SETUP ==========
        self.bridge = CvBridge()
        self.lower_red = np.array([0, 120, 70])
        self.upper_red = np.array([10, 255, 255])
        
        # ========== TTS SETUP ==========
        self.tts_engine = pyttsx3.init()
        self.tts_lock = threading.Lock()
        self.last_spoken = 0
        self.speak_interval = 3.0  # Speak at most every 3 seconds
        
        # ========== ROS SETUP ==========
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 1)
        self.image_pub = self.create_publisher(Image, '/color_detection/output', 1)
        
        # ========== START PROCESSING THREAD ==========
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.get_logger().info("🎯 Simple Color Detection + TTS Ready!")

    def image_callback(self, msg):
        """Just store the latest frame"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.frame_lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().error(f"Camera error: {e}")

    def processing_loop(self):
        """Background processing thread"""
        while self.processing and rclpy.ok():
            # Get frame to process
            frame_to_process = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            
            if frame_to_process is not None:
                try:
                    # Process frame and detect objects
                    processed_frame, object_count = self.detect_objects(frame_to_process)
                    
                    # Publish result
                    self.publish_result(processed_frame)
                    
                    # Speak if objects detected (with rate limiting)
                    current_time = time.time()
                    if object_count > 0 and (current_time - self.last_spoken) > self.speak_interval:
                        self.speak(f"Found {object_count} objects")
                        self.last_spoken = current_time
                    
                    # Display
                    cv2.imshow("Color Detection", processed_frame)
                    cv2.waitKey(1)
                    
                except Exception as e:
                    self.get_logger().error(f"Processing error: {e}")
            
            time.sleep(0.001)  # Small sleep

    def detect_objects(self, frame):
        """Detect colored objects"""
        small = cv2.resize(frame, (640, 480))
        
        # Color detection
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_red, self.upper_red)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        object_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                object_count += 1
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(small, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(small, "Object", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add object count
        cv2.putText(small, f"Objects: {object_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return small, object_count

    def speak(self, text):
        """Speak text in background"""
        def speak_thread():
            with self.tts_lock:
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    self.get_logger().info(f"🗣️ Said: {text}")
                except Exception as e:
                    self.get_logger().error(f"TTS error: {e}")
        
        # Start speaking in background thread
        threading.Thread(target=speak_thread, daemon=True).start()

    def publish_result(self, frame):
        """Publish processed frame"""
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.image_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Publish error: {e}")

    def destroy_node(self):
        self.processing = False
        cv2.destroyAllWindows()
        super().destroy_node()

def main():
    rclpy.init()
    node = SimpleColorTTS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()