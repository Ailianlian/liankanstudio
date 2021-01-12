"""
This file takes care of all the youtube util

Does not use youtube data api for the moment since this is alpha version and not the purpose of the project.
"""

import pytube
from typing import List
import os
import pandas as pd
import numpy as np


# Standard PySceneDetect imports:
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
# For caching detection metrics and saving/loading to a stats file
from scenedetect.stats_manager import StatsManager

# For content-aware scene detection:
from scenedetect.detectors.content_detector import ContentDetector


def find_scenes_static(video_path, automatic=False, threshold=None):
    """
    Find scenes in a video

    Args
    ----

    video_path  : path of the video we want to analyse
    automatic   : analyse the statistics computed to get scenes in order to have the best cut.
    threshold   : if not None, it bypass automatic.

    Returns
    -------

    A list of tuple, each tuple contains with the frame number for begin and end of a scene.

    Note
    ----

    For automatic, we simply use an elbow method. Nothing fancy.
    """
    if threshold is None:
        ret, stat_path = _find_scenes_static(video_path, threshold=30)
        if automatic:
            stat_table = pd.read_csv(stat_path, skiprows=1)
            thresh = _optimize(stat_table["content_val"])
            ret, _ = _find_scenes_static(video_path, threshold=thresh)
    else:
         ret, _ = _find_scenes_static(video_path, threshold=threshold)
        
    return ret

def _get_score_threshold(content_values, threshold, min_scene_length=15):
    """
    Compute a "score" for a given threshold.

    Note
    ----

    Score is computed as the mean of the intraclass variances (mean(std)) and the interclass variance (std(mean))
    We want to maximize the interclass variance and minimize the intraclass variance
    """
    last_cut=0
    pre_mean=[]
    pre_std=[]
    for k,value in enumerate(content_values):
        if value>threshold:
            if k-last_cut>min_scene_length:
                
                pre_mean+=[np.mean(content_values[last_cut:k])]
                pre_std+=[np.std(content_values[last_cut:k])]
                last_cut=k
    intra = np.mean(pre_std)
    inter = np.std(pre_mean)
    return intra, inter

def _get_inter_intra(content_values, min_scene_length=15, threshold_bins=50):
    """
    Compute the intra and inter for a set of threshold

    Args
    ----
    content_values  : the list of value we want to analyse
    min_scene_length: the minimal length of a scene
    threshold_bins  : number of threshold analysed, simple grid search  

    """
    thresholds = np.linspace(np.min(content_values), np.max(content_values)-1, threshold_bins)
    inters = np.zeros(len(thresholds))
    intras = np.zeros(len(thresholds))
    for x, threshold in enumerate(thresholds):
        intras[x], inters[x] = _get_score_threshold(content_values,threshold, min_scene_length)
    return intras, inters, thresholds

def _optimize(content_values,lambda_inter=0.5, min_scene_length=15, bins=70):
    """
    Optimize the score to find optimal threshold for given content_values

    Args
    ----
    content_values  : the list of value we want to analyse
    lambda_inter    : the weight we put on inter score compared to intra score.
    min_scene_length: the minimal length of a scene
    threshold_bins  : number of threshold analysed, simple grid search  

    """
    intras, inters, thresholds= _get_inter_intra(content_values, min_scene_length, threshold_bins=bins)
    return thresholds[np.argmax(inters/max(inters)*lambda_inter+intras/max(intras)*(1-lambda_inter))]

def _find_scenes_static(video_path, threshold):
    """
    Find scenes in a video 

    Args
    ----

    video_path  : path of the video we want to analyse
    threshold   : the higher the less scene you get.

    Returns
    -------

    A list of scene.

    Note
    ----

    Use a python package pyscenedetect, could have been done by hand.
    The algorithm used to detect scene is described here https://pyscenedetect.readthedocs.io/projects/Manual/en/latest/api/detectors.html
    This code can be found at https://pyscenedetect.readthedocs.io/projects/Manual/en/latest/api/scene_manager.html
    We just add the automatic feature, in the true function, seems it exists according to release note just didnt find it in the doc.
    """
    # type: (str) -> List[Tuple[FrameTimecode, FrameTimecode]]
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    # Construct our SceneManager and pass it our StatsManager.
    scene_manager = SceneManager(stats_manager)

    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).
    scene_manager.add_detector(ContentDetector(threshold = threshold))
    base_timecode = video_manager.get_base_timecode()

    # We save our stats file to {VIDEO_PATH}.stats.csv.
    stats_file_path = f'{video_path}.stats.csv'

    scene_list = []

    try:
        # If stats file exists, load it.
        if os.path.exists(stats_file_path):
            # Read stats from CSV file opened in read mode:
            with open(stats_file_path, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file, base_timecode)

        # Set downscale factor to improve processing speed.
        video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list(base_timecode)
        # Each scene is a tuple of (start, end) FrameTimecodes.

        #print('List of scenes obtained:')
        #for i, scene in enumerate(scene_list):
        #    print(
        #        'Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        #        i+1,
        #        scene[0].get_timecode(), scene[0].get_frames(),
        #        scene[1].get_timecode(), scene[1].get_frames(),))

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            with open(stats_file_path, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)

    finally:
        video_manager.release()

    return scene_list, stats_file_path

def parse_url(url: str):
    """
    Parse a youtube url of the form https://www.youtube.com/watch?v={video_id} to get video_id

    Args
    ----

    url : a string of the form https://www.youtube.com/watch?v={video_id}

    Return
    ------

    video_id : the video_id of the youtube video

    """
    # TODO : Add some check to be sure we have a good url
    # TODO : is it good this way of doing it? 
    arg_dict = {elem.split("=")[0]:elem.split("=")[1] for elem in url.split("?")[1].split("&") }
    return arg_dict["v"]


def download(video_id : str, output_dir : str,video_name :str = None, caption_name:str=None, video:bool=True, caption:bool=False, resolution:str="480p", language:List[str]=["fr","a.fr","en","a.en"]):
    """
    Download a youtube video with all the possible surrounding information
    
    Args:
    ----

    video_id : the video_id of the youtube video. Can be found in the URL.
    output_dir : the path where you want to save all the information
    video : True if you want to download the video (480p/mp4 by default)
    caption : True if you want to get the caption of a video if it exists
    resolution : resolution of the video ("480p", "720p" etc). If not set or not available will return a default.
    languages : list of desired language in order of importance
    video_name : 

    # TODO : Separate audio and video? Could be useful.
    """
    _url = f"https://www.youtube.com/watch?v={video_id}"
    _tube = pytube.YouTube(_url)
    if not os.path.isdir(output_dir):
        print('The directory is not present. Creating a new one..')
        os.mkdir(output_dir)
    else:
        print('The directory is present.')
    _stream = _tube.streams.filter(resolution=resolution, file_extension="mp4", only_video=True).first()
    if _stream is None:
        _stream = _tube.streams.filter(file_extension="mp4", only_video=True).first()
    # TODO : test what happen if the video does not exist?
    _video_path = None
    _caption_path = None
    if video:
        _video_path = _stream.download(filename=video_name, output_path=output_dir)
    if caption:
        _caption = None
        k=0
        while _caption is None and k<len(language):
            _caption = _tube.captions.get_by_language_code(language[k])
            k+=1
        _caption_path = _caption.download(caption_name,output_path=output_dir)
    return _video_path, _caption_path
    