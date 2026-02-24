# Graph creation data

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1️⃣ Cloud Upload & Sync Metrics
# -------------------------------
plt.figure(figsize=(6,4))
stages = ['Recording', 'Preprocessing', 'Upload', 'Verification']
avg_latency = [1.8, 2.5, 3.1, 2.2]  # seconds
success_rate = [99, 98, 97, 98.5]  # %

fig, ax1 = plt.subplots(figsize=(6,4))
ax1.bar(stages, avg_latency, color='#60A5FA', alpha=0.7, label='Avg Latency (s)')
ax1.set_ylabel('Latency (s)', color='#60A5FA')
ax1.tick_params(axis='y', labelcolor='#60A5FA')

ax2 = ax1.twinx()
ax2.plot(stages, success_rate, marker='o', color='#4ADE80', linewidth=2, label='Success Rate (%)')
ax2.set_ylabel('Success Rate (%)', color='#4ADE80')
ax2.tick_params(axis='y', labelcolor='#4ADE80')

plt.title('Kairo Cloud Upload & Verification Metrics')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------
# 2️⃣ Model Inference Efficiency (Bar)
# ------------------------------------
plt.figure(figsize=(6,4))
models = ['Whisper', 'Pyannote', 'LangChain LLM', 'Task Detector']
latency = [0.9, 1.4, 2.1, 1.6]
cpu_usage = [42, 55, 68, 49]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, latency, width, label='Latency (s)', color='#FBBF24')
plt.bar(x + width/2, cpu_usage, width, label='CPU Utilization (%)', color='#A78BFA')
plt.xticks(x, models, rotation=15)
plt.ylabel('Performance Metric')
plt.title('Model Inference Efficiency')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ------------------------------------------
# 3️⃣ Battery Efficiency vs Recording Length
# ------------------------------------------
plt.figure(figsize=(6,4))
recording_lengths = [1, 2, 3, 4, 5, 6]  # in hours
battery_usage = [3.2, 6.5, 9.7, 13.0, 16.2, 19.5]  # mAh
efficiency = [96, 94, 92, 89, 87, 85]  # %

fig, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(recording_lengths, battery_usage, color='#F87171', marker='o', linewidth=2, label='Battery Usage (mAh)')
ax1.set_xlabel('Recording Duration (hours)')
ax1.set_ylabel('Battery Usage (mAh)', color='#F87171')
ax1.tick_params(axis='y', labelcolor='#F87171')

ax2 = ax1.twinx()
ax2.plot(recording_lengths, efficiency, color='#22C55E', marker='s', linewidth=2, label='Efficiency (%)')
ax2.set_ylabel('System Efficiency (%)', color='#22C55E')
ax2.tick_params(axis='y', labelcolor='#22C55E')

plt.title('Battery Efficiency vs Recording Duration')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------
# 4️⃣ Semantic Relevance & Contextual Recall
# --------------------------------------------
plt.figure(figsize=(6,4))
weeks = ['W1', 'W2', 'W3', 'W4', 'W5']
semantic_score = [91, 93, 94, 95, 96]
context_recall = [87, 89, 92, 94, 95]

plt.plot(weeks, semantic_score, marker='o', color='#60A5FA', linewidth=2, label='Semantic Relevance (%)')
plt.plot(weeks, context_recall, marker='s', color='#4ADE80', linewidth=2, label='Context Recall (%)')
plt.title('AI Semantic Understanding Progress')
plt.xlabel('Verification Week')
plt.ylabel('Score (%)')
plt.ylim(80, 100)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
