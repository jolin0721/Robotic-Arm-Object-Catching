# Robotic-Arm-Object-Catching
Project done in Media and Cognition Course (fall 2024)

Intelligent Object Sorting System Using Speech Recognition/Text Recognition and Robotic Arm

Project Overview
This project is an intelligent object sorting system that integrates speech recognition, object detection, and robotic arm control to autonomously identify and sort objects into designated bins. The system combines advanced technologies such as Google Speech Recognition API, Faster R-CNN, and OpenAI CLIP to achieve accurate and efficient sorting tasks.

Key Features
Speech Recognition:
Utilizes Google API and SpeechRecognition Library for interpreting voice commands.
Object Detection:
Implements Faster R-CNN for object detection.
Integrates OpenAI CLIP for advanced visual understanding.
Object Classification:
Uses Faster R-CNN with IoU (Intersection over Union) and NMS (Non-Maximum Suppression) for precise classification.
Robotic Arm Control:
A robotic arm identifies, grasps, and places objects into pre-designated bins based on detection results.
Integration Workflow:
Speech command/Text Command → Object detection → Object classification → Robotic arm action.

Technology Stack
Programming Language: Python

Libraries & Tools:
SpeechRecognition
Google Speech API
Faster R-CNN
OpenAI CLIP
PyTorch / TensorFlow

Hardware: Robotic Arm
