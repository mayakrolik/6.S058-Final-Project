import numpy as np
import cv2
# from yt_dlp import YoutubeDL
# from yt_dlp_plugins.extractor.getpot import GetPOTProvider, register_provider
from pytubefix import YouTube
from pytubefix.cli import on_progress

# @register_provider
# class MyProviderRH(GetPOTProvider):
#    _PROVIDER_NAME = 'myprovider'
#    _SUPPORTED_CLIENTS = ('web', )
   
#    def _get_pot(self, client, ydl, visitor_data=None, data_sync_id=None, **kwargs):
#         # Implement your PO Token retrieval here
#         return 'PO_TOKEN'
   
class Scraper():
    """
    pipeline for scraping videos from youtube for computer vision training
    """

    def __init__(self):
        pass

    def load_video(self, video_url):
        opts = {
            'format': 'best',  # Choose the best quality format available
            'extract_audio': True,
            'outtmpl': 'test.%(ext)s'  # Output template for the filename
        }
        with YoutubeDL(opts) as yt:
            yt.download([video_url])
        print(f"Downloaded video: {video_url}")

    def load_audio(self, video_url):
        with YoutubeDL({'format': 'bestaudio'}) as video:
            info_dict = video.extract_info(video_url, download = True)
            video_title = info_dict['title']
            print(video_title)

    def separate_audio_video(self):
        pass

    def extract_transcript_given(self):
        pass

    def extract_transcript_model(self):
        pass

class Scraper2():

    def __init__(self):
        pass

    def extract_urls_from_playlist(self, playist_url):
        """
        Given a youtube playlist, returns a list of the individual video URLs

        returns: list of str urls
        """
        pass

    def load_video(self, file):
        """
        given a file path, returns the video object at that file
        """
        pass

    def load_audio(self, file):
        """
        given a file paht, returns the audio object at that file
        """
        pass

    def scrape_video_and_audio(self, url, resolution='1080p', file_prefix = None):
        """
        Given a url, downloads both the video and audio (separate files) to
        raw_videos

        returns: none
        """
        yt = YouTube(url, on_progress_callback=on_progress)

        video = yt.streams.filter(resolution='1080p').first()
        if video:
            print('Downloading Video...')
            video.download(filename=f'raw_videos/{file_prefix if file_prefix else yt.title}_video.mp4')
        else:
            pass

        audio = yt.streams.filter(only_audio=True).first()
        if audio:
            print('Downloading Audio...')
            audio.download(filename=f'raw_videos/{file_prefix if file_prefix else yt.title}_audio.mp4')
        else:
            pass


if __name__ == "__main__":
    scrap = Scraper2()
    scrap.scrape_video_and_audio("https://youtu.be/dQw4w9WgXcQ?si=IehRXHFWcJMhFQy5")
    