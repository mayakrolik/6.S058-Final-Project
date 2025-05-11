import os
from scraper import Scraper
from cropper import Cropper, save2vid
from moviepy import VideoFileClip
from pydub import AudioSegment
import math
import shutil
import csv
import glob
import whisper
import subprocess



def scrape_from_playlist_url(playlist_url, output_dir = "raw_videos"):
    """
    Handles all the scraping for you!
    """

    scrap = Scraper()
    urls = scrap.get_all_urls_from_playlist(playlist_url)
    
    skipped = 0
    i = 0
    total_len = len(urls)
    for url in urls:
        sucess = scrap.scrape_video_and_audio(url, file_prefix=f"raw_{i}", output_path=output_dir)
        if not sucess:
            skipped += 1
        else: # increment because sucessful download
            i += 1

    print(f"Scraping complete for playlist {playlist_url}, successfully scrapped {total_len-skipped} out of {total_len} videos {(total_len-skipped)/total_len}")

def chop_up_found_files(dir, save_dir, final_size = 30):
    """
    given a directory of scrapped videos, chop up the videos into final_size
    second long clips
    """
    counter = 0
    meta_data = []
    dir_path = os.path.dirname(os.path.realpath(__file__))

    for file in glob.glob(f"{dir}/raw_*_video.mp4"):
        i = file.split("_")[-2]
        vid_path = f"{dir}/raw_{i}_video.mp4"
        aud_path = f"{dir_path}/{dir}/raw_{i}_audio.mp3"
        vid = VideoFileClip(vid_path)
        aud = AudioSegment.from_file(aud_path)
        duration = vid.duration

        if duration <= final_size:
            print("video is undersized, still kept")
            shutil.copy(vid_path, f"{dir_path}/{save_dir}/clip_{counter}_video.mp4")
            shutil.copy(aud_path, f"{dir_path}/{save_dir}/clip_{counter}_audio.mp3")
            counter += 1
            continue


        for j in range(int(duration//final_size)):
            clip_vid = vid.subclipped(j*final_size, (j+1)*final_size)
            clip_aud = aud[j*final_size* 1000: (j+1)*final_size* 1000]

            clip_vid.write_videofile(f"{dir_path}/{save_dir}/clip_{counter}_video.mp4")
            clip_aud.export(f"{dir_path}/{save_dir}/clip_{counter}_audio.mp3", format="mp3")
            counter += 1

            # save metadata
            inner = [counter, f"{save_dir}/clip_{counter}_video.mp4", f"{save_dir}/clip_{counter}_audio.mp3"]
            meta_data.append(inner)


    with open('meta.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(meta_data)
    
    return meta_data

def track_lips_and_transcript(intermed_dir, destination_dir):
    """
    get transcripts and crop to lip motion of all videos
    """
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    cropper = Cropper()
    model = whisper.load_model("base")
    print("loaded models")
    files = glob.glob(f"{intermed_dir}/clip_*_video.mp4")
    total_num = len(files)

    failed_files = []
    errors = []

    for j, file in enumerate(files):
        i = file.split("_")[-2]
        print(f"Processing clip {i}, {j/total_num} done, {j}th file of {total_num}")
        vid_path = f"{intermed_dir}/clip_{i}_video.mp4"
        aud_path = f"{intermed_dir}/clip_{i}_audio.mp3"

        # get the transcript!
        transcript = model.transcribe(aud_path)
        with open(f"{destination_dir}/clip_{i}_transcript.txt", "w") as file:
            file.write(transcript["text"])

        # crop the vid
        try:
            post_vid = cropper(vid_path)
            save2vid(f"{destination_dir}/clip_{i}_lips.mp4", post_vid, frames_per_second=30)
        except Exception as e:
            failed_files.append(vid_path)
            errors.append(e)

        # get a single frame, arbitrarily set to the 2nd second for funzies :p
        vid = VideoFileClip(vid_path)
        vid.save_frame(f"{destination_dir}/clip_{i}_frame.png", t = 2)

    print(failed_files)
    print(errors)
    return failed_files

def run_demographics_analysis(final_data_dir):
    """
    Runs inference models to label the data for demographic characteristics
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    to_csv = [["img_path"]]
    for file in glob.glob(f"{final_data_dir}/clip_*_frame.png"):
        to_csv.append([f"{dir_path}/{file}"])

    with open("file_paths.csv", "w") as f:
        write = csv.writer(f)
        write.writerows(to_csv)
    
    subprocess.run(["python", "models/predict.py", "--csv", "file_paths.csv"])

def pipeline(playlist_url):
    """
    Given a youtube playlist, runs the whole sha-bang
    """

    # scrape_from_playlist_url(playlist_url)

    # chop_up_found_files("raw_videos", "intermediate_videos")

    # track_lips_and_transcript("intermediate_videos", "processed_videos")

    run_demographics_analysis("processed_videos")

if __name__ == "__main__":
    pipeline("https://www.youtube.com/playlist?list=PL4A014bThEmoxMp35yjj56U1kWy5PEoYH")
    # aud = AudioSegment.from_file("/Users/mayakrolik/code/6.S058/6.S058 Final Project/raw_videos/raw_0_audio.mp3")
    
    # /Users/mayakrolik/code/6.S058/6.S058 Final Project/raw_videos/raw_46_audio.mp3
    # res = aud[:10*1000]
    # res.export("/Users/mayakrolik/code/6.S058/6.S058 Final Project/exported.mp3", format = "mp3")