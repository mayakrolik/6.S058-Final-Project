from scraper import Scraper
from cropper import Cropper


def scrape_from_playlist_url(playlist_url):
    """
    Handles all the scraping for you!
    """

    scrap = Scraper()
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

def pipeline(playlist_url):
    """
    Given a youtube playlist, runs the whole sha-bang
    """

    scrape_from_playlist_url(playlist_url)

if __name__ == "__main__":
    pipeline("https://www.youtube.com/playlist?list=PL4A014bThEmoxMp35yjj56U1kWy5PEoYH")