# UR10e Voice-Controlled Robotic Manipulation System

### 🎥 Demo Video
https://www.youtube.com/watch?v=rlMUFu1V42M

This repository contains a **voice-controlled robotic manipulation framework** for a **Universal Robots UR10e** platform. The system integrates **large language models, computer vision, RGB-D perception, and robotic control** to translate natural language commands into safe and executable robotic actions.

The system allows users to issue spoken commands such as:

> "Pick up the red cylinder and place it on the right."

The system then interprets the command, detects objects in the environment, and generates the required robot motion commands.

---

# Features

• Voice-based robot command interface using **speech recognition and Whisper**  
• Natural language reasoning using **OpenAI GPT models**  
• Real-time perception using **Intel RealSense RGB-D camera**  
• **ArUco marker calibration** for camera–robot coordinate alignment  
• Object detection using **color segmentation and shape recognition**  
• Autonomous planning for tasks including:
  - object picking
  - stacking
  - sorting
  - arranging in lines
  - arranging in circles
• Safety-aware robot motion with **workspace constraints**
• Interactive **GUI control interface**

---

# System Architecture

The system integrates multiple components:
Voice Command
│
Speech Recognition (Whisper)
│
Natural Language Reasoning (LLM)
│
Task Planning
│
Computer Vision (RealSense + OpenCV)
│
Object Localization
│
Robot Motion Execution (UR10e)


---

# Hardware Requirements

- **Universal Robots UR10e**
- **Intel RealSense depth camera**
- Gripper compatible with UR robot
- Calibration **ArUco markers**
- Workstation running Python

---

# Software Dependencies

Python 3.9+ recommended.

Required packages include:
openai
numpy
opencv-python
pyrealsense2
speechrecognition
pyaudio
tkinter
URBasic


Install core dependencies:

```bash
pip install openai numpy opencv-python speechrecognition pyaudio

Repository Structure
.
├── main.py
├── SupportCodes/
│   └── GripperFunctions.py
├── URBasic/
├── README.md
└── .gitignore

Setup
1. Clone the repository
git clone https://github.com/yourusername/ur10e-voice-controller.git
cd ur10e-voice-controller
2. Set OpenAI API key

Set your API key as an environment variable.

Linux / Mac:

export OPENAI_API_KEY="your_key_here"

Windows:

setx OPENAI_API_KEY "your_key_here"
3. Connect hardware

Ensure the following devices are connected:

• UR10e robot
• Intel RealSense camera
• Robot gripper

Update the robot IP in the script if needed.

4. Run the system
python main.py

The GUI will open and allow voice-based robot control.

Example Commands

Examples of supported voice commands:

• "Pick up the red cylinder."
• "Stack the blocks."
• "Place the objects in a line."
• "Arrange the cylinders in a circle."
• "Move the robot home."

Research Use

This repository was developed for robotics research and experimentation in human–robot interaction and language-guided manipulation.

Citation
To be updated

Disclaimer

This software is provided for research and educational purposes.
Use at your own risk when operating robotic hardware.


---

### What you should do now
1. Create the repo  
2. Select **MIT License**  
3. Turn **Add README → ON**  
4. Paste this content

---

If you'd like, I can also give you a **much stronger academic version of the README** (the kind used in top robotics labs like Stanford / Berkeley / CMU).
