import threading
import time
import cv2
import json
import pyaudio
import os
from eyefeature import EyeTrackingMouse
from smile import run_smile_control
from head import run_head_control
from calibration_manager import CalibrationManager

# Try importing Vosk (offline)
VOSK_AVAILABLE = False
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    print("⚠️ Vosk not installed")

# Import SpeechRecognition as backup (online)
SPEECH_REC_AVAILABLE = False
try:
    import speech_recognition as sr
    SPEECH_REC_AVAILABLE = True
except ImportError:
    print("⚠️ SpeechRecognition not installed")

def listen_with_vosk():
    """
    STRICT OFFLINE voice recognition - single command trigger
    """
    print("\n⏳ Loading Vosk model (OFFLINE)...")
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "model")

        print(f"📂 Model path: {MODEL_PATH}")

        if not os.path.exists(MODEL_PATH):
            print("❌ Model folder NOT FOUND")
            return None

        model = Model(MODEL_PATH)

    except Exception as e:
        print(f"❌ Vosk model error: {e}")
        return None
    
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)
    
    # Set grammar to ONLY recognize our three words
    grammar = '["eye", "smile", "head", "[unk]"]'
    rec.SetGrammar(grammar)
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=pyaudio.paInt16, 
            channels=1, 
            rate=16000, 
            input=True,
            frames_per_buffer=8000
        )
        stream.start_stream()
    except Exception as e:
        print(f"❌ Microphone error: {e}")
        return None
    
    print("✅ Vosk Voice System Ready (OFFLINE)")
    print("✅ Grammar-constrained recognition")
    print("🔊 Listening... (Say commands CLEARLY)\n")
    
    VALID_COMMANDS = {
        'eye': 0,
        'smile': 1,
        'head': 2
    }
    
    last_command_time = 0
    cooldown = 2.5  # 2.5 second cooldown between commands
    
    while True:
        try:
            data = stream.read(8000, exception_on_overflow=False)
            
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get('text', '').lower().strip()
                
                # Only process if enough time has passed
                if not text or time.time() - last_command_time < cooldown:
                    continue
                
                # Check for valid command
                detected_command = None
                
                # Check exact phrase first
                if text in VALID_COMMANDS:
                    detected_command = text
                else:
                    # Check individual words
                    words = text.split()
                    for word in words:
                        if word in VALID_COMMANDS:
                            detected_command = word
                            break
                
                # If we detected a command, trigger it immediately
                if detected_command:
                    mode_idx = VALID_COMMANDS[detected_command]
                    print(f"✅ COMMAND: {detected_command.upper()}\n")
                    last_command_time = time.time()
                    yield mode_idx
                
        except Exception as e:
            continue

def listen_with_google():
    """
    STRICT ONLINE voice recognition using Google Web Speech API
    """
    print("\n✅ Using Google Web Speech API (ONLINE)")
    print("🔊 Listening... (Say commands CLEARLY)\n")
    
    recognizer = sr.Recognizer()
    
    # Higher threshold to reduce false triggers
    recognizer.energy_threshold = 5000
    recognizer.dynamic_energy_threshold = False
    recognizer.pause_threshold = 0.8
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.5
    
    microphone = sr.Microphone()
    
    # Calibrate
    print("🎤 Calibrating for ambient noise (stay QUIET for 3 seconds)...")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=3)
    print(f"✅ Calibration complete! (Threshold: {recognizer.energy_threshold})\n")
    
    VALID_COMMANDS = {
        'eye': 0,
        'smile': 1,
        'head': 2
    }
    
    last_command_time = 0
    cooldown = 2.5  # 2.5 second cooldown
    
    while True:
        try:
            with microphone as source:
                print("🎤 Ready for command...")
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=2)
            
            try:
                text = recognizer.recognize_google(
                    audio, 
                    language='en-US',
                    show_all=False
                ).lower().strip()
                
                # Only process if enough time has passed
                if time.time() - last_command_time < cooldown:
                    print(f"⏳ Cooldown active ({cooldown - (time.time() - last_command_time):.1f}s remaining)")
                    continue
                
                print(f"🗣️ Heard: '{text}'")
                
                # Check for valid command
                detected_command = None
                
                # Check exact phrase first
                if text in VALID_COMMANDS:
                    detected_command = text
                else:
                    # Check individual words
                    words = text.split()
                    for word in words:
                        if word in VALID_COMMANDS:
                            detected_command = word
                            break
                
                # If we detected a command, trigger it immediately
                if detected_command:
                    mode_idx = VALID_COMMANDS[detected_command]
                    print(f"✅ COMMAND: {detected_command.upper()}\n")
                    last_command_time = time.time()
                    yield mode_idx
                else:
                    print("❓ Not a valid command - ignored\n")
            
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"❌ Google API error: {e}")
                time.sleep(2)
                continue
                
        except sr.WaitTimeoutError:
            continue
        except Exception as e:
            continue

def listen_for_commands():
    """
    Smart voice command listener with automatic fallback
    """
    print("\n🎤 Initializing Voice Recognition System...")
    print("⚠️ IMPORTANT: Wait 2.5 seconds between commands")
    
    # Try Vosk first (offline)
    if VOSK_AVAILABLE:
        print("📡 Attempting OFFLINE recognition (Vosk)...")
        vosk_generator = listen_with_vosk()
        if vosk_generator is not None:
            yield from vosk_generator
            return
        else:
            print("❌ Vosk failed to initialize")
    
    # Fallback to Google Speech API (online)
    if SPEECH_REC_AVAILABLE:
        print("📡 Falling back to ONLINE recognition (Google)...")
        print("⚠️ Requires internet connection!")
        time.sleep(1)
        yield from listen_with_google()
        return
    
    # If both fail
    print("\n" + "="*60)
    print("❌ NO VOICE RECOGNITION AVAILABLE")
    print("="*60)
    print("Install: pip install vosk")
    print("    or: pip install SpeechRecognition")
    print("="*60)
    
    while True:
        yield None
        time.sleep(1)

def run_feature(mode, cap, stop_flag, calibration_manager):
    """Run the selected feature mode"""
    modes = [
        "👁️ Eye Tracking",
        "😊 Smile Tab Switch",
        "🔄 Head Scroll"
    ]
    
    print(f"\n{'='*50}")
    print(f"🚀 ACTIVATED: {modes[mode]}")
    print(f"{'='*50}\n")
    
    if mode == 0:
        eye = EyeTrackingMouse()
        eye.run_modular(cap, stop_flag)
    elif mode == 1:
        run_smile_control(cap, stop_flag, calibration_manager)
    elif mode == 2:
        run_head_control(cap, stop_flag, calibration_manager)
    
    print(f"\n⏹️ STOPPED: {modes[mode]}\n")
    print("🔊 Listening for commands...\n")

def auto_calibrate_smile(cap, calibration_manager):
    """Calibrate smile using lip corner distance"""
    print("\n📸 Smile Calibration Starting...")
    print("⚠️ Keep NEUTRAL face (no smile) for 3 seconds!")
    
    import mediapipe as mp
    import math
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    LEFT_LIP_CORNER = 61
    RIGHT_LIP_CORNER = 291
    
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Calibration", 100, 100)
    cv2.resizeWindow("Calibration", 640, 480)
    
    # Collect neutral samples
    neutral_samples = []
    start_time = time.time()
    
    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_corner = landmarks[LEFT_LIP_CORNER]
            right_corner = landmarks[RIGHT_LIP_CORNER]
            
            x1 = int(left_corner.x * w)
            y1 = int(left_corner.y * h)
            x2 = int(right_corner.x * w)
            y2 = int(right_corner.y * h)
            
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            neutral_samples.append(distance)
            
            cv2.circle(frame, (x1, y1), 5, (0, 255, 255), -1)
            cv2.circle(frame, (x2, y2), 5, (0, 255, 255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            cv2.putText(frame, f"Distance: {distance:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        remaining = int(3 - (time.time() - start_time))
        cv2.putText(frame, f"NEUTRAL FACE: {remaining}s", (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        cv2.imshow("Calibration", frame)
        cv2.waitKey(1)
    
    neutral_distance = sum(neutral_samples) / len(neutral_samples) if neutral_samples else 50.0
    print(f"✅ Neutral distance: {neutral_distance:.1f}")
    
    # Collect smile samples
    print("Now SMILE BIG for 3 seconds!")
    time.sleep(0.5)
    
    smile_samples = []
    start_time = time.time()
    
    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_corner = landmarks[LEFT_LIP_CORNER]
            right_corner = landmarks[RIGHT_LIP_CORNER]
            
            x1 = int(left_corner.x * w)
            y1 = int(left_corner.y * h)
            x2 = int(right_corner.x * w)
            y2 = int(right_corner.y * h)
            
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            smile_samples.append(distance)
            
            cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.putText(frame, f"Distance: {distance:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        remaining = int(3 - (time.time() - start_time))
        cv2.putText(frame, f"SMILE BIG: {remaining}s", (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        cv2.imshow("Calibration", frame)
        cv2.waitKey(1)
    
    smile_distance = sum(smile_samples) / len(smile_samples) if smile_samples else neutral_distance * 1.2
    cv2.destroyAllWindows()
    
    # Calculate threshold
    diff = smile_distance - neutral_distance
    smile_threshold = diff * 0.6
    
    if smile_threshold < 5:
        smile_threshold = 8
    
    calibration_manager.set_smile_calibration(neutral_distance, smile_threshold)
    
    print(f"✅ Smile distance: {smile_distance:.1f}")
    print(f"✅ Difference: {diff:.1f}")
    print(f"✅ Threshold: {smile_threshold:.1f}")
    print(f"✅ Will trigger when distance > {neutral_distance + smile_threshold:.1f}\n")

def auto_calibrate_head(cap, calibration_manager):
    """Automatically calibrate head"""
    print("\n📸 Head Calibration Starting...")
    print("Hold head STEADY for 3 seconds!")
    
    import mediapipe as mp
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Calibration", 100, 100)
    cv2.resizeWindow("Calibration", 640, 480)
    
    neutral_samples = []
    start_time = time.time()
    
    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            nose = face_landmarks.landmark[1]
            nose_y = int(nose.y * h)
            neutral_samples.append(nose_y)
            
            cv2.circle(frame, (int(nose.x * w), nose_y), 6, (0, 255, 0), -1)
        
        remaining = int(3 - (time.time() - start_time))
        cv2.putText(frame, f"HOLD STEADY: {remaining}s", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        cv2.imshow("Calibration", frame)
        cv2.waitKey(1)
    
    neutral_y = sum(neutral_samples) / len(neutral_samples) if neutral_samples else 240
    cv2.destroyAllWindows()
    
    calibration_manager.set_head_calibration(neutral_y)
    print(f"✅ Head neutral: {neutral_y:.1f}\n")

def initial_calibration(cap, calibration_manager):
    """Perform automatic initial calibration"""
    print("\n" + "="*60)
    print("🎯 AUTOMATIC CALIBRATION")
    print("="*60)
    
    if not calibration_manager.is_calibrated('smile'):
        auto_calibrate_smile(cap, calibration_manager)
    else:
        cal = calibration_manager.get_smile_calibration()
        print(f"✅ Smile already calibrated:")
        print(f"   Neutral: {cal['neutral_intensity']:.1f}")
        print(f"   Threshold: {cal['smile_threshold']:.1f}")
        print(f"   Trigger at: {cal['neutral_intensity'] + cal['smile_threshold']:.1f}\n")
    
    if not calibration_manager.is_calibrated('head'):
        auto_calibrate_head(cap, calibration_manager)
    else:
        print("✅ Head already calibrated\n")
    
    print("="*60)
    print("✅ CALIBRATION COMPLETE!")
    print("="*60)
    print("\n🎤 Say: 'EYE', 'SMILE', or 'HEAD'\n")

if __name__ == "__main__":
    print("="*60)
    print("🎯 FACE CONTROL SYSTEM")
    print("="*60)
    print("\n📋 Features:")
    print("   • 'EYE' → Eye tracking + double blink to click")
    print("   • 'SMILE' → Smile to switch tabs")
    print("   • 'HEAD' → Head tilt to scroll")
    print("\nControls: Q = Stop | Ctrl+C = Exit")
    print("="*60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        exit()
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("\n✅ Webcam ready!")
    
    calibration_manager = CalibrationManager()
    initial_calibration(cap, calibration_manager)
    
    current_mode = None
    stop_flag = None
    thread = None
    
    try:
        for new_mode in listen_for_commands():
            if new_mode is None:
                continue
            
            if current_mode == new_mode:
                print(f"ℹ️ Already running. Press Q to stop first.")
                continue
            
            if thread is not None and thread.is_alive():
                print(f"\n⏸️ Stopping...")
                stop_flag.set()
                thread.join(timeout=3)
                time.sleep(0.5)
            
            current_mode = new_mode
            stop_flag = threading.Event()
            thread = threading.Thread(target=run_feature, args=(current_mode, cap, stop_flag, calibration_manager))
            thread.daemon = True
            thread.start()
            time.sleep(0.3)
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Shutting down...")
        if stop_flag:
            stop_flag.set()
        if thread:
            thread.join(timeout=2)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Goodbye!\n")
