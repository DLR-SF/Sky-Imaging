# Documentation of the image upload via FTP-S

## Sky imager (client)

### Set up
Bring the ASI in a network in which the ports for FTP-S upload are open.

ASI firmware should be updated to MX-V4.7.3.11-r4 (2021-04-21) or newer.

Log on to the camera.

Go to Admin Menu > FTP profiles

Set the server settings as shown in the screenshot below.

In the server settings set server ip address (and port) to 212.170.96.95:990.

Set user name to pyranocam

Set password (request from niklas.blum@dlr.de or michael.meinel@dlr.de).

Set connection to passive FTP.

Set the FTP profile etc. (to be defined). For a test, you can use the configuration shown in the image below. Make sure that the server directory name matches with the folder structure present on the server.
<img src="media_ftp_upload/FTPprofile.PNG" alt="FTP profile used for the connection test" title="FTP profile used for the connection test" width="600"/>

### Test

Go to Admin Menu > Test Current Network Configuration

Scroll to Image Transfer and the profile which you activated in the FTP profile e.g. *FTP-Webcam* (see image).

<img src="media_ftp_upload/screenshotTestFTPprofile.PNG" alt="Test Current Network Configuration window" title="Test Current Network Configuration window" width="600"/>

Click transfer. A log file will appear. The following log file is received in the case of success:

<img src="media_ftp_upload/screenshotTestFTPprofileLog.PNG" alt="log file of the connection test" title="log file of the connection test" width="600"/>


## Server

The server is located in Paseo office (old knecht 4).

The server allows to connect via FTP and FTP-S. However, if the client connects via FTP it will change its protocol to FTP-S before log in or fail to connect if this is not possible.



## Target machine
On the machine which processes the images, the pyranocam repositories should be cloned.

Install the following
`conda create -n datacheck python=3.8´
`conda activate datacheck´
`conda install -c conda-forge pythonnet´

Set up an automatic task:
Open Windows task scheduler
Go to task library
Create nw task...
	General:
		Name: PyranoCam sync
		Set Run independently from user login
	Trigger:
		New trigger: Daily, at 00:01
	Actions:
		Start program, program path "C:\git\pyranocam\evaluations\daq_and_transfer\run_autosync.bat"
		
		
	Click OK and enter user password
