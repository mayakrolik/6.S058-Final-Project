from scraper import Scraper2 as Scraper
from cropper import Cropper


def pipeline(playlist_url):
    """
    Given a youtube playlist, runs the whole sha-bang
    """

    scrap = Scraper()
    crop = Cropper()

    video_urls = scrap.extract_urls_from_playlist(playlist_url)
    num_videos = len(video_urls)

    file_name = "test1"
    for i, url in enumerate(video_urls):
        scrap.scrape_video_and_audio(url, file_prefix=f"{file_name}_{i}")

    for i in range(num_videos):
        video = scrap.load_video(f"raw_videos/{file_name}_{i}_video.mp4")
        audio = scrap.load_audio(f"raw_videos/{file_name}_{i}_audio.mp4")

        pruned_video, pruned_audio = crop.cut_lipless_frames(video, audio)
        crop.crop_to_lips(pruned_video)