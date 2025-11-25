# Documentation on the setup of the SkyCam System

## Quality check

(Should best be done before giving the camera to the user)

Set up the camera outside take some sun images. Best if the atmosphere is very clear. Check if any abnormal flare effects are seen. (Scratches, faulty coating, ...)


## Create a site-specific config file

(Should best be done before giving the camera to the user)

Use the following file as basis:
`<path_to_repository>\data\camera_data\mobotix_cfg_templates\`

Adapt the following:
- Make sure text printed to image is short enough so that it does not reach into exposed image area leave some safety margin.
- If you already know the local network settings. Adapt these in the config file:
  - Set a hostname, so you will find the camera, replace: `<<<<HOSTNAME>>>>` (e.g. `Cloud_Cam_xxx`). We recommend to use the camera id here
  - Set IP for camera in the local network, replace: `<<<<CAM_IP>>>>` (e.g. `10.21.202.146`)
  - Set the subnetmask of your local network, replace: `<<<<SUBNETMASK>>>>` (e.g. `255.255.255.224`)
  - Set gateway (usually the router's IP), replace: `<<<<GATEWAY>>>>` (e.g. `10.21.202.158`)
  - Set dns servers, replace: `<<<<DNS_SERVERS>>>>` (e.g. `172.21.154.193 129.247.160.1 8.8.8.8`). The last IP is public and therefore a safe option.
- Specify the URL (or IP) of the sftp server to which you might want to upload replacing the tag `<<<<FTP_URL>>>>`.
- FTP folder to store to, if possible use the camera name. Try to use a consistent camera name everywhere, replace: `<<<<CAM_FOLDER>>>>`  (e.g. `Cloud_Cam_Golden`). We recommend to use the camera id here.
- Login to the (s)ftp server may require a user name and password. Replace the tags `<<<<FTP_USER>>>>` and `<<<<FTP_PW>>>>` accordingly.
- Camera time zone, replace  `<<<<TIMEZONE>>>>` (e.g. `GMT/GMT+1`)
- Set NTP servers which can be reached by the camera, replace `<<<<NTP_SERVER>>>>` (e.g. `1.de.pool.ntp.org 0.de.pool.ntp.org`)

Consider changing the following:

Section FTP is configured to upload to our server, this section may be adapted for some projects.
```
SECTION ftp
...
ENDSECTION ftp
```

If you recognize the need, update the template and commit it.


## Adapt the ASI's network settings to your local network
Find out the camera's IP address by pressing the button on the camera. It will read out loud the camera's IP address etc.

Configure your PC's network settings to be able to connect with the camera. Then enter the camera's web interface and assign an IP address, etc. according to you local network. Then store the current configuration and reboot the camera.

Adapt your PC's network settings to be able to connect with the camera to test if everything worked fine. If not go back to the start of this section. 

Connect the camera at its final location to the network. Connect your computer to the same network. Make sure you can access the camera's web interface.


## Install the camera physically
Mount the camera horizontally. Make sure the camera is installed at a similar height as sensors nearby so that obstructions are avoided in any sensor's field of view. Orient the camera so that the upper image side will be south (camera should be marked accordingly by manufacturer). In case of the mobotix Q26 the camera is marked with top (see image). Top marks the top of the 360° panoramic view.

<img src="media_asi_setup/q26Top.PNG" alt="Schematic view on the Q26 camera from above." title="" width="300"/>

If possible, mount the camera's bracket on a tripod and level it horizontally. Make sure the leveling of the bracket is secured sufficiently to be stable over time. Always use 2 nuts on counter.


## Update the camera firmware

(Should best be done before giving the camera to the user)

Open the camera's web interface. For this, type the cameras IP in a browser's address bar.

In the upper right image corner you should find an info button (i). Press it and check the firmware version given under "Software", (e.g. MX-V4.7.3.11)

If your firmware has a smaller number than MX-V4.7.3.11 (Q25) or MX-V5.2.1.4 (Q26), update it to at least that version as follows.

Download the desired firmware from mobotix.com. 

Open the admin menu in the camera's web interface.

Go to Update system software.

Upload the firmware to the camera.

Start the firmware update.


## Upload the configuration file

(Should best be done before giving the camera to the user)

Open the cameras web interface. For this, type the cameras IP in a browser's address bar.

Go to admin ...

Under configuration select "Load configuration from local computer"

With Browse select the provided config file for the station.

Select "Replace everything except the parts checked below".

Untick all sections. Consider setting a tick near Networking. In this case you are sure not to lose the connection to the camera by 
unintentionally changing these settings. Only if you untick this section, configuration parameters regarding the camera's host name,
IP address, subnet mask, gateway, DNI server,... will be replaced which will be desired in many cases.

Press upload.

In the next page. Press "store" 

Then "store permanently".

Then "Reboot".

Then "Reboot now".

The camera will restart with the new configuration using the previous IP address.



## Configure your local network
If the camera cannot upload its images to the FTP server yet, you may need to configure your local network. 

The outgoing ports 990 and 1337 to 1374 in the firewall in your network need to be open for the upload.



## Download the final configuration file

Open the camera's web interface. For this, type the cameras IP in a browser's address bar.

Go to the admin menu

Under “configuration” select "Save current configuration to local computer"

You will download a .cfg file. 

Store the cfg file in your documentation.

