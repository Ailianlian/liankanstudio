import click 
# DOES NOT WORK AS DIRECT FUNCTION
from .util.yt_util import download, parse_url
from .capture import Capture
from .detector import Detector
import os

@click.group()
def liankan():
    pass

@liankan.command()
@click.option('--url/--no-url', default=True, help="Tells if this is or not a url. No means this is a video_id")
@click.option('--res', default="144p", help="Resolution, can be 140p, 240p, 480p, 720p")
@click.argument('video_input')
@click.argument('video_output')
def dl(url, video_input, video_output, res):
    """Download the video_input into video_output\n
    
    Args\n
    ----\n

    video_input : url or video_id depending on url option\n
    video_output: output_path of the video
    """
    output_dir = os.path.dirname(video_output)
    filename = os.path.basename(video_output).split('.')[0]
    if url:
        download(parse_url(video_input),output_dir=output_dir, video_name=filename, resolution=res)
    else:
        download(video_input,output_dir=output_dir,video_name=filename, resolution=res)

@liankan.command()
@click.option('--thresh', default=0.5, help="Threshold for confidence of box.")
@click.option('--rate', default=10, help="Do detection every rate frames in the video.")
@click.option('--tracker', default="csrt", help="Tracker type.")
@click.option('--method', default="yolo", help="Detection method.")
@click.option('--target', default="person", help="The object we want to detect in image.")
@click.option('--verbose/--no-verbose', default=True, help="If True, output information about the processing.")
@click.option('--cache/--no-cache', default=True, help="If True, register file to do a warm restart of the processing.")
@click.option('--look/--no-look', default=True, help="Show the processed video at the end of processing if True")
@click.argument('input_path')
@click.argument('output_path')
def detect(input_path, output_path, look, cache, verbose, rate, tracker, method, target, thresh):
    """Download the video_input into video_output\n
    
    Args\n
    ----\n

    video_input : url or video_id depending on url option\n
    video_output: output_path of the video
    """
    capture = Capture(input_path)
    capture.find_scenes()
    detector = Detector(method, target=target)
    capture.compute_stat(detector, rate, scenes=None, tracker_type="csrt", verbose=verbose, output_path=None, identifier=None, max_frame=-1,
    threshold_detect=thresh, merge_thresh=0.3, cpu_count=None, video_path=output_path, look_result=look, cache=cache, release_end=True)

if __name__ == '__main__':
    # USELESS IT BUGS :D
    liankan()