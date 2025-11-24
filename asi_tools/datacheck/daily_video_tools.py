# Source: https://pythonexamples.org/python-opencv-cv2-create-video-from-images/
# git_test
# Modules and Packages
import cv2
from datetime import date
from datetime import datetime, timedelta
import sys
import pytz
import logging
from _datetime import datetime, timedelta
import glob
import os
import pvlib
from calendar import monthrange
import dateutil.tz
import pandas as pd
import numpy as np
import re

def create_video(site, evaluatedDate, pathImageStorage='//129.247.24.131/Meteo/MeteoCamera',
                 exp_time_range=[0, int(1e10)]):
           
    
    yesterdate = evaluatedDate.strftime("%Y/%m/%d")
    chosen_date = evaluatedDate.strftime("%Y_%m_%d")
    path_date = yesterdate

    # Timestemp for Europe/Berlin UTC +2
    tz = pytz.timezone('Europe/Berlin')
    now = datetime.now(tz)
    current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    filename_path_date = evaluatedDate.strftime("%Y%m%d") 
    
    #Creating AVI-File
    newpath = f"{pathImageStorage}/{path_date[0:4]}/Cloud_Cam_{site}/{path_date[5:]}/000_AVI"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    i = 0
    hour = 0
    num_of_frames = 0
    out = None

    print("The programme is running. Please wait until the video file has been created...")   
    while i < 24:  # Loop to collect all images from the several subfolders
        i += 1
        if i < 10:
            hour = "0" + str(i)
        else:
            hour = str(i)

        path = f"{pathImageStorage}/{path_date[0:4]}/Cloud_Cam_{site}/{path_date[5:]}/{hour}/{filename_path_date}*.jpg"
        # In "path" {hour} will be replaced by the variable "hour" from the while-loop
        for filename in glob.glob(path):
            try:
                exp_time_found = int(re.search(r"_(\d{1,9}).", filename).groups()[0])
            except Exception as e:
                logging.error(f'File: {filename} has invalid exposure time.')
                continue
            if not (exp_time_range[0] <= exp_time_found <= exp_time_range[1]):
                continue

            img = cv2.imread(filename)
            cloudcam_pic_height,cloudcam_pic_width,_ = np.shape(img)
            img = cv2.resize(img, (int(1024*cloudcam_pic_width/cloudcam_pic_height), 1024))
            cloudcam_pic_height,cloudcam_pic_width,_ = np.shape(img)
            if os.path.getsize(filename) == 0:
                #print("Error: Images corrupted. File Size is 0kB.")
                logging.error(f'File: {filename} has size 0kB.')
            
            if not type(out) == cv2.VideoWriter:
                # Imageproperties
                
                frameSize = (cloudcam_pic_width, cloudcam_pic_height)
                out = cv2.VideoWriter(
                    f"{pathImageStorage}/{path_date[0:4]}/Cloud_Cam_{site}/{path_date[5:]}/000_AVI/raw.avi",
                    cv2.VideoWriter_fourcc(*'DIVX'), 25, frameSize)
            out.write(img)
            num_of_frames += 1

    out.release()

    if os.path.getsize(
            f"{pathImageStorage}/{path_date[0:4]}/Cloud_Cam_{site}/{path_date[5:]}/000_AVI/raw.avi") < 100000:
        #print("Error: Video corrupted. Check Image-Files")
        logging.warning('Size of Video is < 100kB')

        exit()

    # Renaming the file
    old_file = os.path.join(
        f"{pathImageStorage}/{path_date[0:4]}/Cloud_Cam_{site}/{path_date[5:]}/000_AVI/raw.avi")
    new_file = os.path.join(
        f"{pathImageStorage}/{path_date[0:4]}/Cloud_Cam_{site}/{path_date[5:]}/000_AVI/Cloud_Cam_{site}_{chosen_date}_{current_time}_{num_of_frames}.avi")
    os.rename(old_file, new_file)

    print(f"Video file was succesfully created.\n\nPath: {newpath}")

if __name__ == "__main__":

    logging.basicConfig(filename=f'log_daily_video_{datetime.now():%Y%m%d_%H%M%S}.txt', level=logging.INFO)

    # predefining name variables (otherwise error in fstring)
    site = "siteerror"
    chosen_date = "dateerror"
    year = "yearerror"
    month = "montherror"
    day = "dayerror"
    path_date = "pathdateerror"
    filename_path_date = "filenamepathdateerror"

    # defining valid arg options
    months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
              "november", "december"]

    # Defining the arguments via "sys.argv"
    if len(sys.argv) == 1:
        print(
            "At least one argument expected.\n\nPositional Arguments:\n\nSite (Patras, Benguerir, Murdoch, Golden)\n\nOptional Arguments:\n\nyear (e.g. '2022')\n\nmonth (e.g. '04' or 'April')\n\nday (e.g. '05')")
        exit()
    elif len(sys.argv) > 1:
        site = str(sys.argv[1])

    if len(sys.argv) == 2:
        today = date.today()
        evaluatedDate = today - timedelta(days=1)
    elif len(sys.argv) > 2:
        year = str(sys.argv[2])
        month = str(sys.argv[3]).lower()
        day = str(sys.argv[4])
        if len(year) != 4:
            print("Argument 'year' must consist of 4 digits (e.g. 2022)")
            exit()
        if month in months:
            if months.index(month) < 9:
                month = str(months.index(month) + 1)
                month = "0" + str(month)
            elif months.index(month) >= 9:
                month = str(months.index(month) + 1)
            elif len(month) != 2:
                print(
                    "Argument 'month' must consist of 2 digits (e.g. 04 for 'April'). Name of the month is also accepted (e.g. 'April')")
                exit()
            elif int(month) > 12:
                print(
                    "Argument 'month' must consist of 2 digits (e.g. 04 for 'April'). Name of the month is also accepted (e.g. 'April')")
                exit()
        if len(day) != 2:
            print("Argument 'day' must consist of 2 digits (e.g. 04 for the '4th of...')")
            exit()
            
        evaluatedDate = datetime.strptime(f'{year}-{month}-{day}','%Y-%m-%d').date()
   
    create_video(site, evaluatedDate)