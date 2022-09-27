# PSim-RCAM
Python implementation of the non-linear 6DOF GARTEUR RCAM aircraft flight dynamics model.

Group for Aeronautical Research and Technology Europe (GARTEUR) - Research Civil Aircraft Model (RCAM).
http://garteur.org/wp-content/reports/FM/FM_AG-08_TP-088-3.pdf

The excellent tutorials by Christopher Lum (for Matlab/Simulink) were used as guides:
<p>
1 - Equations/Modeling: https://www.youtube.com/watch?v=bFFAL9lI2IQ
<p>
2 - Matlab implementation: https://www.youtube.com/watch?v=m5sEln5bWuM

The program runs the integration loop at a user defined frame-rate, adjusting the integration steps to the available computing cycles to render real-time data to FlightGear.

Output is sent to FlightGear (FG), over UDP, at a user specified frame rate.
The FG interface uses the class implemented by Andrew Tridgel: 
<p>
fgFDM: https://github.com/ArduPilot/pymavlink/blob/master/fgFDM.py

Currently, the UDP address is set to the local machine.

Run this program in one terminal and from a separate terminal, start FG with one of the following commands (depending on the aircraft addons installed):

fgfs --airport=KSFO --runway=28R --aircraft=ufo --native-fdm=socket,in,60,,5500,udp --fdm=null

fgfs --airport=KSFO --runway=28R --aircraft=Embraer170 --aircraft-dir=./FlightGear/Aircraft/E-jet-family/ --native-fdm=socket,in,60,,5500,udp --fdm=null

fgfs --airport=KSFO --runway=28R --aircraft=757-200-RB211 --aircraft-dir=~/.fgfs/Aircraft/org.flightgear.fgaddon.stable_2020/Aircraft/757-200 --native-fdm=socket,in,60,,5500,udp --fdm=null

fgfs --airport=KSFO --runway=28R --aircraft=757-200-RB211 --aircraft-dir=~/.fgfs/Aircraft/org.flightgear.fgaddon.stable_2020/Aircraft/757-200 --native-fdm=socket,in,60,,5500,udp --fdm=null --enable-hud --turbulence=0.5 --in-air  --enable-rembrandt

REQUIRES a joystick to work. Tested with Logitech USB Stick.
