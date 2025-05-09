import numpy as np
import cv2
from pytubefix import YouTube, Playlist, Search
from pytubefix.cli import on_progress
from pytubefix.exceptions import VideoUnavailable, MaxRetriesExceeded


class Scraper():

    def __init__(self):
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

    def get_videos_from_search(self, query = 'Github Issue Best Practices'):
        """"""
        results = Search(query)

        # for video in results.videos:
        #     print(f'Title: {video.title}')
        #     print(f'URL: {video.watch_url}')
        #     print(f'Duration: {video.length} sec')
        #     print('---')

        print(type(results.videos[0]))
        return results.videos

    def get_all_urls_from_playlist(self, playlist_url):
        """
        Given a string youtube playlist returns all the urls of each video
        """
        return Playlist(playlist_url)

    def scrape_audio_only(self, url, file_prefix=None):
        """
        Given a url, downloads audio

        returns: none
        """
        yt = YouTube(url, on_progress_callback=on_progress)

        audio = yt.streams.filter(only_audio=True).first()
        if audio:
            print('Downloading Audio...')
            audio.download(output_path= "raw_videos", filename=f'{file_prefix if file_prefix else yt.title}_audio.mp3')
        else:
            pass

    # def apply_to_playlist(self, playlist_url, function, *func_args):
    #     """
    #     given a playlist, execute a certain function to each
    #     passed in function must take in a url as its first parameter
    #     """
    #     play = self.get_all_urls_from_playlist(playlist_url)
    #     for i, video in enumerate(play):
    #         function(video, *func_args)

    def scrape_video_and_audio(self, url, resolution='1080p', file_prefix = None, output_path = "raw_videos"):
        """
        Given a url, downloads both the video and audio (separate files) to
        raw_videos

        returns: bool successful download
        """
        try:
            yt = YouTube(url, on_progress_callback=on_progress)
        # except (VideoUnavailable, MaxRetriesExceeded):
        except:
            print(f"video at {url} is unavalible, skipping :)")
            return False

        else:
            video = yt.streams.filter(resolution='1080p').first()
            audio = yt.streams.filter(only_audio=True).first()
            if video and audio:
                # print('Downloading Video...')
                video.download(output_path= output_path, filename=f'{file_prefix if file_prefix else yt.title}_video.mp4')
                # print('Downloading Audio...')
                audio.download(output_path= output_path, filename=f'{file_prefix if file_prefix else yt.title}_audio.mp3')
                return True
            else:
                print(f"Either the audio or video was un scrape-able for {url}, skipping :)")
                return False


if __name__ == "__main__":
    scrap = Scraper()
    playlist_url = "https://www.youtube.com/playlist?list=PL4A014bThEmoxMp35yjj56U1kWy5PEoYH"
    urls = scrap.get_all_urls_from_playlist(playlist_url)
    
    skipped = 0
    i = 0
    total_len = len(urls)
    for url in urls:
        sucess = scrap.scrape_video_and_audio(url, file_prefix=f"raw_{i}")
        if not sucess:
            skipped += 1
        else: # increment because sucessful download
            i += 1

    print(f"Scraping complete for playlist {playlist_url}, successfully scrapped {total_len-skipped} out of {total_len} videos {(total_len-skipped)/total_len}")
    # scrap.apply_to_playlist("https://www.youtube.com/watch?v=WxCz0UNUNeQ&list=OLAK5uy_mV1rxm7sOPx1S8YKoiKKlZcJ2GBDXY4cs", scrap.scrape_audio_only, output_path = "fun")
    # scrap.scrape_video_and_audio("https://youtu.be/fxFIqIp5eQo", file_prefix=None)
    # scrap.get_videos_from_search()
    