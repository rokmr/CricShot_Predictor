import streamlit as st
import subprocess
import os
import shutil
import transformers
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from torch import nn


# All paths
FFMPEG_PATH = shutil.which("ffmpeg")
# Required dictionary
idx_to_class = {0: 'DEFENSE', 1: 'LOFTED', 2: 'SQUARE CUT', 3: 'SWEEP'}
class_label_mapping = {'DEFENSE': 0, 'LOFTED': 1, 'SQUARE CUT': 2, 'SWEEP': 3}

cuurent_dir = os.getcwd()
print(f"Current Directory: {cuurent_dir}")

# Definig the paths 
MODEL_PATH = "./cricketshot/cricket.pt"
saved_model_path = './cricketshot/model'
saved_processor_path = './cricketshot/processor'

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the clip model and processor for generating embeddings
@st.cache_data()
def embeddings_creators():
    processor = CLIPProcessor.from_pretrained(saved_processor_path)
    clip_model = CLIPModel.from_pretrained(saved_model_path)
    clip_model.to(device)
    # Load your model here

    return processor, clip_model

processor, clip_model = embeddings_creators()

# Define the LSTM Network for sequence data classification
class LSTMNetwork(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_classes=4):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use the output of the last time step
        return x
    
# Load the model and its weights
@st.cache_data()
def load_model():
    # Load your model here
    model = LSTMNetwork(input_size=512, hidden_size=256, num_classes=len(class_label_mapping)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    return model

model = load_model()





frames_dir = "./demo/frames/"
os.makedirs(frames_dir, exist_ok=True)


# Device Agnostic
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Streamlit App
st.image("https://fanplayiot.com/wp-content/uploads/2024/01/01_fanplay_logo_240.png")
st.title('FanPlay IoT')
st.header("Please upload the cricket shot video")


# video input
video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
if video_file is None:
    st.stop()
    st.write("Please upload video file.")
else:    
    st.write(f"Video File: {video_file.name}")

# Specify the directory to save the video
save_directory = './demo'
video_path = os.path.join(save_directory, video_file.name)
with open(video_path, "wb") as f:
    f.write(video_file.getbuffer())

# video conversion to mp4
def convert_to_mp4(input_file, output_dir="./demo"):
    """Converts a video file to MP4 using FFmpeg.

    Args:
        input_file (str): Path to the input video file.
        output_dir (str, optional): Directory to save the output MP4. 
                                    Defaults to the current directory.
    """

    filename, file_ext = os.path.splitext(input_file)
    output_file = filename + ".mp4"

    command = [
        "ffmpeg",
        "-i", input_file,
        output_file
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Conversion successful: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        return False

new_video_path = os.path.join(save_directory, f"{os.path.splitext(video_file.name)[0]}.mp4")

# Convert the video to MP4 format
filename, file_ext = os.path.splitext(video_file.name)
if file_ext.lower() not in [".mp4"]:
    if convert_to_mp4(video_path):
        st.write("Video converted to MP4 format.")
        os.remove(video_path)
    else:
        st.write("Failed to convert video to MP4 format.")


st.video(new_video_path)

# Extract frames from the video
output_pattern = os.path.join(frames_dir, "video_frame_%04d.jpg")
ffmpeg_command = f"{FFMPEG_PATH} -i {new_video_path} -vf fps=1 {output_pattern} -loglevel quiet"

try:
    subprocess.run(ffmpeg_command, shell=True, check=True)
    print(f"Processed {video_file}.")
except subprocess.CalledProcessError:
    print(f"Failed to process video: {new_video_path}")
os.remove(new_video_path)

# Extract the frames paths and loading the frames
image_paths = []
for root, dirs, files in os.walk(frames_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_paths.append(os.path.join(root, file))

inference_images = [Image.open(path).convert("RGB") for path in image_paths]

# creating embeddings
tokens = processor(text=None, images=inference_images, return_tensors="pt").to(device)
inference_embeddings = clip_model.get_image_features(**tokens)




# Prediction
with torch.no_grad():
    output = model(inference_embeddings.unsqueeze(0))
    idx = output.argmax()
    prob = output.softmax(dim=1)
    # st.write(f"confidence: {torch.max(prob):.4f}")
    st.write(f" Prediction: {idx_to_class[idx.item()]}")
    st.progress(int(torch.max(prob) * 100), "Confidence")

# Delete the frames directory
try:
    shutil.rmtree(frames_dir)
    print(f"Folder '{frames_dir}' and its contents have been deleted.")
except Exception as ess:
    print(f"Error while deleting folder '{frames_dir}': {e}")
