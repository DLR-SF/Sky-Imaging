# Creates a daily video for a camera with exposure series
import sys
from asi_tools.datacheck.daily_video_tools import *

logging.basicConfig(filename='log_daily_video', level=logging.INFO)

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
    print("At least one argument expected.\n\nPositional Arguments:\n\nSite (Patras, Benguerir, Murdoch, Golden)"
        "\n\nOptional Arguments:\n\nyear (e.g. '2022')\n\nmonth (e.g. '04' or 'April')\n\nday (e.g. '05')")
    exit()
elif len(sys.argv) > 1:
    site = str(sys.argv[1])

if len(sys.argv) == 2:
    exp_time_range = [120, 180]
elif len(sys.argv) > 2:
    exp_time_range=[int(sys.argv[2]), int(sys.argv[3])]


if len(sys.argv) <= 4:
    today = date.today()
    evaluatedDate = today - timedelta(days=1)
elif len(sys.argv) > 4:
    year = str(sys.argv[4])
    month = str(sys.argv[5]).lower()
    day = str(sys.argv[6])
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

    evaluatedDate = datetime.strptime(f'{year}-{month}-{day}', '%Y-%m-%d').date()

create_video(site, evaluatedDate, exp_time_range=exp_time_range)