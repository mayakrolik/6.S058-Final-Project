import argparse
import math
import os
import pickle
import shutil
import warnings

import ffmpeg
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from utils import save_vid_aud

warnings.filterwarnings("ignore")

def process_video_audio(args, vid_filename, aud_filename):
    if args.landmarks_dir:
        landmarks_filename = (
            vid_filename.replace(args.vid_dir, args.landmarks_dir)[:-4] + ".pkl"
        )
        landmarks = pickle.load(open(landmarks_filename, "rb"))
    else:
        landmarks = None
        
    try:
        video_data = vid_dataloader.load_data(vid_filename, landmarks)
        audio_data = aud_dataloader.load_data(aud_filename)
    except (UnboundLocalError, TypeError, OverflowError, AssertionError):
        return False
        
    if video_data is None:
        return False

    # Process segments
    for i, start_idx in enumerate(range(0, len(video_data), seg_vid_len)):
        dst_vid_filename = (
            f"{vid_filename.replace(args.vid_dir, dst_vid_dir)[:-4]}_{i:02d}.mp4"
        )
        dst_aud_filename = (
            f"{aud_filename.replace(args.aud_dir, dst_vid_dir)[:-4]}_{i:02d}.wav"
        )
        trim_video_data = video_data[start_idx : start_idx + seg_vid_len]
        trim_audio_data = audio_data[
            :, start_idx * 640 : (start_idx + seg_vid_len) * 640
        ]
        if trim_video_data is None or trim_audio_data is None:
            continue
            
        video_length = len(trim_video_data)
        audio_length = trim_audio_data.size(1)
        if (
            audio_length / video_length < 560.0
            or audio_length / video_length > 720.0
            or video_length < 12
        ):
            continue

        # Save video and audio
        save_vid_aud(
            dst_vid_filename,
            dst_aud_filename,
            trim_video_data,
            trim_audio_data,
            video_fps=25,
            audio_sample_rate=16000,
        )

        # Merge video and audio
        if args.combine_av:
            in1 = ffmpeg.input(dst_vid_filename)
            in2 = ffmpeg.input(dst_aud_filename)
            out = ffmpeg.output(
                in1["v"],
                in2["a"],
                dst_vid_filename[:-4] + ".m.mp4",
                vcodec="copy",
                acodec="aac",
                strict="experimental",
                loglevel="panic",
            )
            out.run()
            os.remove(dst_aud_filename)
            os.remove(dst_vid_filename)
            shutil.move(dst_vid_filename[:-4] + ".m.mp4", dst_vid_filename)
    
    return True

# Argument parsing
parser = argparse.ArgumentParser(description="Video/Audio Preprocessing")
parser.add_argument(
    "--vid-file",
    type=str,
    help="Path to the input video file",
)
parser.add_argument(
    "--aud-file",
    type=str,
    help="Path to the input audio file",
)
parser.add_argument(
    "--vid-dir",
    type=str,
    help="Directory where video files are stored (alternative to --vid-file)",
)
parser.add_argument(
    "--aud-dir",
    type=str,
    help="Directory where audio files are stored (alternative to --aud-file)",
)
parser.add_argument(
    "--label-dir",
    type=str,
    default="",
    help="Directory where labels are saved (if processing multiple files)",
)
parser.add_argument(
    "--landmarks-dir",
    type=str,
    default=None,
    help="Directory of landmarks",
)
parser.add_argument(
    "--detector",
    type=str,
    help="Type of face detector",
)
parser.add_argument(
    "--root-dir",
    type=str,
    required=True,
    help="Root directory for preprocessed output",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="custom",
    help="Name of dataset",
)
parser.add_argument(
    "--seg-duration",
    type=int,
    default=16,
    help="Max duration (second) for each segment, (Default: 16)",
)
parser.add_argument(
    "--combine-av",
    type=lambda x: (str(x).lower() == "true"),
    default=False,
    help="Merge audio and video components into a single media file",
)
parser.add_argument(
    "--groups",
    type=int,
    default=1,
    help="Number of threads for parallel processing",
)
parser.add_argument(
    "--job-index",
    type=int,
    default=0,
    help="Job index for parallel processing",
)
args = parser.parse_args()

# Constants
seg_vid_len = args.seg_duration * 25
seg_aud_len = args.seg_duration * 16000
dst_vid_dir = os.path.join(
    args.root_dir, args.dataset, f"{args.dataset}_video_seg{args.seg_duration}s"
)
os.makedirs(dst_vid_dir, exist_ok=True)

# Initialize data loaders
vid_dataloader = AVSRDataLoader(
    modality="video", detector=args.detector, convert_gray=False
)
aud_dataloader = AVSRDataLoader(modality="audio")

# Process single file or multiple files
if args.vid_file and args.aud_file:
    # Process single video/audio pair
    success = process_video_audio(args, args.vid_file, args.aud_file)
    if not success:
        print(f"Failed to process {args.vid_file}")
elif args.vid_dir and args.aud_dir and args.label_dir:
    # Process multiple files from directories
    filenames = [
        os.path.join(args.vid_dir, _ + ".mp4")
        for _ in open(os.path.join(args.label_dir, "vox-en.id")).read().splitlines()
    ]
    
    unit = math.ceil(len(filenames) / args.groups)
    files_to_process = filenames[args.job_index * unit : (args.job_index + 1) * unit]

    for vid_filename in tqdm(files_to_process):
        aud_filename = vid_filename.replace(args.vid_dir, args.aud_dir)[:-4] + ".wav"
        success = process_video_audio(args, vid_filename, aud_filename)
        if not success:
            print(f"Failed to process {vid_filename}")
else:
    raise ValueError("You must specify either --vid-file/--aud-file OR --vid-dir/--aud-dir/--label-dir")