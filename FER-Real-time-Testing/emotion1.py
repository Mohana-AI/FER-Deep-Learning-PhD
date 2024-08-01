import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict
import numpy as np
import csv

# Initialize counters for each engagement state
engaged_count = 0
neutral_count = 0
disengaged_count = 0

# Track engagement over time
engagement_over_time = defaultdict(lambda: [0, 0, 0])
time_step = 0

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Set up the figure and bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Collect data for CSV
csv_data = []

# Function to update the plot
def update_chart(frame):
    global engaged_count, neutral_count, disengaged_count, time_step

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        return

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Get emotion predictions
        emotions = result[0]['emotion']

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Classify engagement
        if emotion in ['happy', 'surprise', 'fear'] and emotions[emotion] > 80:
            engagement = 'Highly engaged'
            engaged_count += 1
        elif emotion == 'neutral' and emotions['neutral'] > 50:
            engagement = 'Engaged'
            neutral_count += 1
        else:
            engagement = 'Disengaged'
            disengaged_count += 1

        # Update engagement over time
        engagement_over_time[time_step][0] += 1 if engagement == 'Highly engaged' else 0
        engagement_over_time[time_step][1] += 1 if engagement == 'Engaged' else 0
        engagement_over_time[time_step][2] += 1 if engagement == 'Disengaged' else 0

        # Collect data for CSV
        csv_data.append([time_step, emotion, emotions[emotion], engagement])

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f'{emotion} ({emotions[emotion]:.2f}%)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, engagement, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display emotion percentages on the side
        y0, dy = 30, 30
        for i, (emo, perc) in enumerate(emotions.items()):
            text = f'{emo}: {perc:.2f}%'
            cv2.putText(frame, text, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Update the chart every second (approx)
    if time_step % 10 == 0:
        ax1.clear()
        ax1.set_title('Engagement Visualization')
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Student Engagements')
        times = sorted(engagement_over_time.keys())
        engaged = [engagement_over_time[t][0] for t in times]
        neutral = [engagement_over_time[t][1] for t in times]
        disengaged = [engagement_over_time[t][2] for t in times]
        ax1.stackplot(times, disengaged, neutral, engaged, labels=['Disengaged', 'Engaged', 'Highly Engaged'], colors=['blue', 'orange', 'green'])
        ax1.legend(loc='upper left')

        ax2.clear()
        ax2.set_title('Visualization for Student')
        total_counts = engaged_count + neutral_count + disengaged_count
        ax2.pie([disengaged_count, neutral_count, engaged_count], labels=['Disengaged', 'Engaged', 'Highly Engaged'], autopct='%1.2f%%', colors=['blue', 'orange', 'green'])

    # Increment time step
    time_step += 1

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

        # Save the figures before closing
        plt.savefig('engagement_chart.png')

        # Write data to CSV file
        with open('engagement_data.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Time Step', 'Dominant Emotion', 'Emotion Percentage', 'Engagement'])
            csvwriter.writerows(csv_data)

        plt.close(fig)
        return

# Update the chart every 100 milliseconds
ani = FuncAnimation(fig, update_chart, interval=100)

# Show the chart
plt.show()