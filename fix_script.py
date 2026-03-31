with open("vrlFace/face/recognizer.py", "r") as f:
    content = f.read()
    
# Find the end of face_search function correctly
print(content[-500:])
