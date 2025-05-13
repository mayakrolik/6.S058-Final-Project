import os
from scraper import Scraper
from cropper import Cropper, save2vid
from moviepy import VideoFileClip
from llm_prompting import call_chat, iterative_prompt, initial_promt
from metric import compute_cross_category_scores, plot_heatmap
from pydub import AudioSegment
import pandas as pd
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


def scrape_from_search_query(query, top_n = 10, output_dir = "raw_videos", seen_before = None, indx = 0):
    """
    Handles all the scraping for you given a youtube search and a limit of videos to try
    """

    scrap = Scraper()
    raw_urls = iter(scrap.get_videos_from_search(query))

    urls = set()
    if not seen_before:
        seen_before = set()

    while len(urls) <= top_n:
        to_consider = next(raw_urls)
        if to_consider not in seen_before:
            urls.add(to_consider)
    
    skipped = 0
    total_len = len(urls)
    scrapped = []
    for url in urls:
        sucess = scrap.scrape_video_and_audio(url, file_prefix=f"raw_{indx}", output_path=output_dir)
        if not sucess:
            skipped += 1
        else: # increment because sucessful download
            scrapped.append(url)
            indx += 1

    print(f"Scraping complete for search {query}, successfully scrapped {total_len-skipped} out of {total_len} videos {(total_len-skipped)/total_len}")
    return urls, indx

def chop_up_found_files(dir, save_dir, final_size = 30, counter = 0):
    """
    given a directory of scrapped videos, chop up the videos into final_size
    second long clips
    """
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


        for j in range(min(int(duration//final_size), 6)):
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
    
    return meta_data, counter

def track_lips_and_transcript(intermed_dir, destination_dir, crop_lips = True):
    """
    get transcripts and crop to lip motion of all videos
    """
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    cropper = Cropper()
    model = whisper.load_model("base")
    print("loaded models")
    # existing_files = glob.glob(f"{destination_dir}/clip_*_video.mp4")
    # files = glob.glob(f"{intermed_dir}/clip_*_video.mp4")
    # to_process = list(set(files) - set(existing_files))
    # total_num = len(files)

    existing_files = glob.glob(f"{destination_dir}/clip_*_lips.mp4")
    existing_files = {str_path.replace("lips", "video") for str_path in existing_files}
    existing_files = {str_path.replace(destination_dir, intermed_dir) for str_path in existing_files}

    files = glob.glob(f"{intermed_dir}/clip_*_video.mp4")
    # files = {str_path.split("_")[-2] for str_path in files}

    to_process = list(set(files) - existing_files)
    total_num = len(to_process)

    failed_files = []
    errors = []

    for j, file in enumerate(to_process):
        i = file.split("_")[-2]
        print(f"\nProcessing clip {i}, {j/total_num} done, {j}th file of {total_num}\n")
        vid_path = f"{intermed_dir}/clip_{i}_video.mp4"
        aud_path = f"{intermed_dir}/clip_{i}_audio.mp3"

        # get the transcript!
        transcript = model.transcribe(aud_path)
        with open(f"{destination_dir}/clip_{i}_transcript.txt", "w") as file:
            file.write(transcript["text"])

        # crop the vid
        if crop_lips:
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

def run_demographics_analysis_kaggle_dat():
    """
    https://www.kaggle.com/datasets/apoorvwatsky/miraclvc1?resource=download
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))

    to_csv = [["img_path"]]
    for file in glob.glob("archive/dataset/dataset/*/*/*/*/color_001.jpg"):
        to_csv.append([f"{dir_path}/{file}"])

    with open("llm_run.csv", "w") as f:
        write = csv.writer(f)
        write.writerows(to_csv)
    
    subprocess.run(["python", "models/predict.py", "--csv", "llm_run.csv"])

def run_demographics_analysis(final_data_dir, iteration_num, output_file_name = "output.csv", save_data_file = "demographics.csv"):
    """
    Runs inference models to label the data for demographic characteristics
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    to_csv = [["img_path"]]
    for file in glob.glob(f"{final_data_dir}/clip_*_frame.png"):
        to_csv.append([f"{dir_path}/{file}"])

    with open(output_file_name, "w") as f:
        write = csv.writer(f)
        write.writerows(to_csv)
    
    subprocess.run(["python", "models/predict.py", "--csv", output_file_name])

    prev = pd.read_csv(save_data_file)
    new = pd.read_csv("test_outputs.csv")
    combo = pd.concat([prev, new], ignore_index=True)
    merged_df = combo.drop_duplicates(subset=["face_name_align"], keep='first')
    merged_df.to_csv(save_data_file, index=False)
    merged_df.to_csv(f"demographics_{iteration_num}.csv")

def pipeline(playlist_url):
    """
    Given a youtube playlist, runs the whole sha-bang
    """

    # scrape_from_playlist_url(playlist_url)

    # chop_up_found_files("raw_videos", "intermediate_videos")

    # track_lips_and_transcript("intermediate_videos", "processed_videos")

    run_demographics_analysis("processed_videos")

def llm_pipeline(purpose, n_iters = 6):
    """
    The whole shap-bang, but smart
    """
    visited_urls = set()
    scores = []
    query = call_chat(iterative_prompt(purpose, "black senior males"))
    raw_dir = "Attempt_3/raw_videos"
    intermed_dir = "Attempt_3/intermediate_videos"
    processed_dir = "Attempt_3/processed_videos"
    csv_path = "demographics.csv"

    AGE_LUMPING = {
    '3-9': 'Child and Young Adult',
    '10-19': 'Child and Young Adult',
    '20-29': 'Child and Young Adult',
    '30-39': 'Adult',
    '40-49': 'Adult',
    '50-59': 'Senior',
    '60-69': 'Senior',
    '70+': 'Senior'
}

    RACE_LUMPING = {
        'White': 'White',
        'Black': 'Black',
        'Latino_Hispanic': 'Latino_Hispanic',
        'East Asian': 'Asian',
        'Southeast Asian': 'Asian',
        'Indian': 'Asian',
        'Middle Eastern': 'White'
    }

    ALL_CATEGORIES = {
    'race_lumped': ['White', 'Black', 'Latino_Hispanic', 'Asian'],
    'gender': ['Male', 'Female'],
    'age_group_lumped': ['Child and Young Adult', 'Adult', 'Senior']
}
    T_LOW_COVERAGE = 0.2
    indx = 0
    count = 0

    for i in range(n_iters):
        print(f"\nBEGINNING ITERATION {i} with query: {query}")
        new_urls, indx = scrape_from_search_query(query, seen_before=visited_urls, output_dir=raw_dir, indx=indx, top_n=5)
        visited_urls = visited_urls | new_urls
        error_msgs, count = chop_up_found_files(raw_dir, intermed_dir, counter=count)
        track_lips_and_transcript(intermed_dir, processed_dir)
        run_demographics_analysis(processed_dir, i)
        df = pd.read_csv(csv_path)
        df['age_group_lumped'] = df['age'].map(AGE_LUMPING)
        df['race_lumped'] = df['race'].map(RACE_LUMPING)
        coverage_score, per_group_scores, low_coverage_groups = compute_cross_category_scores(df, ALL_CATEGORIES, T_LOW_COVERAGE)

        # select group of interest
        sorted_low = sorted(low_coverage_groups.items(), key=lambda x: x[1])
        group, score = sorted_low[0]
        group_name = group.replace("-", " ")
        scores.append(coverage_score)
        print(f"SELECTED GROUP {group_name} with score {score}, overall score sofar is {coverage_score}")
        query = call_chat(iterative_prompt(purpose, group_name))

        if i == n_iters - 1:
            plot_heatmap(per_group_scores, ALL_CATEGORIES)

    print(coverage_score, scores)
    return coverage_score, scores


if __name__ == "__main__":
    # pipeline("https://www.youtube.com/playlist?list=PL4A014bThEmoxMp35yjj56U1kWy5PEoYH")

    detection_errors = """Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_519_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_755_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_788_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_607_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_11_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_816_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_760_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_626_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_410_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_189_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_542_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_774_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_387_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_208_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_758_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_791_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_785_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_736_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_247_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_521_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_229_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_779_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_409_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_766_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_544_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_772_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_291_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_687_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_459_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_747_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_88_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_639_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_549_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_828_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_104_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_348_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_676_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_783_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_789_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_740_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_160_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_754_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_424_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_775_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_96_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_319_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_411_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_761_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_557_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_784_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_737_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_723_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_209_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_759_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_790_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_191_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_778_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_699_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_686_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_371_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_767_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_598_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_746_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_752_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_564_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_704_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_183_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_7_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_731_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_61_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_105_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_663_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_762_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_624_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_630_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_540_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_776_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_427_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_757_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_575_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_743_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_69_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_13_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_838_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_715_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_471_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_584_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_608_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_787_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_734_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_751_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_421_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_171_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_486_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_603_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_308_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_764_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_552_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_546_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_732_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_596_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_769_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_541_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_777_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_631_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_763_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_604_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_574_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_742_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_481_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_12_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_756_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_426_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_50_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_652_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_786_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_579_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_609_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_735_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_667_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_792_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_487_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_744_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_328_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_602_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_799_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_750_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_493_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_401_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_771_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_221_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_684_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_529_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_286_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_765_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_749_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_780_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_583_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_712_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_418_frame.png'
Sorry, there were no faces found in '/Users/mayakrolik/code/6.S058/6.S058 Final Project/processed_videos/clip_768_frame.png'
    """
    
    
    # aud = AudioSegment.from_file("/Users/mayakrolik/code/6.S058/6.S058 Final Project/raw_videos/raw_0_audio.mp3")
    
    # run_demographics_analysis_kaggle_dat()

    llm_pipeline("recognizing lip motion in English", n_iters=5)
#     purpose = "recognizing lip motion in English"
#     visited_urls = set()
#     scores = []
#     # query = call_chat(initial_promt(purpose))
#     raw_dir = "raw_llm_videos"
#     intermed_dir = "intermediate_llm_videos"
#     processed_dir = "processed_llm_videos"
#     csv_path = "demographics.csv"

#     AGE_LUMPING = {
#     '3-9': 'Child and Young Adult',
#     '10-19': 'Child and Young Adult',
#     '20-29': 'Child and Young Adult',
#     '30-39': 'Adult',
#     '40-49': 'Adult',
#     '50-59': 'Senior',
#     '60-69': 'Senior',
#     '70+': 'Senior'
# }

#     RACE_LUMPING = {
#         'White': 'White',
#         'Black': 'Black',
#         'Latino_Hispanic': 'Latino_Hispanic',
#         'East Asian': 'Asian',
#         'Southeast Asian': 'Asian',
#         'Indian': 'Asian',
#         'Middle Eastern': 'White'
#     }

#     ALL_CATEGORIES = {
#     'race_lumped': ['White', 'Black', 'Latino_Hispanic', 'Asian'],
#     'gender': ['Male', 'Female'],
#     'age_group_lumped': ['Child and Young Adult', 'Adult', 'Senior']
# }
#     T_LOW_COVERAGE = 0.2
#     indx = 0
#     count = 0

#     run_demographics_analysis(processed_dir)
#     df = pd.read_csv(csv_path)
#     df['age_group_lumped'] = df['age'].map(AGE_LUMPING)
#     df['race_lumped'] = df['race'].map(RACE_LUMPING)
#     coverage_score, per_group_scores, low_coverage_groups = compute_cross_category_scores(df, ALL_CATEGORIES, T_LOW_COVERAGE)

#     # select group of interest
#     sorted_low = sorted(low_coverage_groups.items(), key=lambda x: x[1])
#     group, score = sorted_low[0]
#     group_name = group.replace("-", " ")
#     scores.append(coverage_score)
#     print(f"SELECTED GROUP {group_name} with score {score}, overall score sofar is {coverage_score}")
#     query = iterative_prompt(purpose, group_name)
#     print(f"new query: {query}")

#     plot_heatmap(per_group_scores, ALL_CATEGORIES)
    