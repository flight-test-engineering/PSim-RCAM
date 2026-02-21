<img width="3840" height="2160" alt="from_below" src="https://github.com/user-attachments/assets/f1e18d71-cb84-44b5-a1de-5bc535ad681e" />

# PSim-RCAM
Welcome to PSim-RCAM - short for Python Simulation - Research Civil Aircraft Model!

This is a Python implementation of the non-linear flight dynamics model published by:  
Group for Aeronautical Research and Technology Europe (GARTEUR) - [Research Civil Aircraft Model (RCAM) (rev Jun 1995)](http://garteur.org/wp-content/reports/FM/FM_AG-08_TP-088-3.pdf)  
HOWEVER:  
    # a few equations and values are only available in the later RCAM revision (dated Feb 1997)  
    # the 1997 revision is not easy to find  
    # the values and equations from this reference were obtained from youtube videos listed below:  

The excellent tutorials by Christopher Lum (for Matlab/Simulink) were used as a guide:  
1 - [Equations/Modeling](https://www.youtube.com/watch?v=bFFAL9lI2IQ)  
2 - [Matlab implementation](https://www.youtube.com/watch?v=m5sEln5bWuM)  

In addition to what prof. Lum implements, the following features are added here:  
1 - Ground reactions (landing gear), to allow for takeoff and landing  
2 - Actuator dynamics  
3 - Turbofan engine deck (with parallel processing/multi-core), based on: [Turbofan Palylist](https://www.youtube.com/playlist?list=PLqJt-rNo8TZ1Y5Pzlk4S1205NUt2t-t22)  
4 - Terrain database based in SRTM data, if FlightGear not available  
5 - Terrain database from FlightGear  
6 - FlightGear visualization via UDP (with parallel threading)  
7 - Ground effect on CL and CD  

## Module:
The code is split in module files for better organization:  
```
.
|-- main.py - the main simulation loop and control
|-- ISA_module.py - International Standard Atmosphere (see * below)
|-- rcam_parameters.json - configuration file with aircraft data (for the RCAM model)
|-- psim
|   |-- physics.py - core differential and non-differential equations
|   |-- propulsion.py - turbofan engine deck functions (see ** below)
|   |-- PW2000_similar_deck.csv - engine cycle deck for a PW2000 class engine (**)
|   |-- environment.py - speed, course and geodesy
|   |-- constants.py - indexes for controls, states and other constants non-aircraft model related
|   |-- config.py - json aircraft configuration parser
|   |-- io
|       |-- fgFDM.py - FlightGear packet construction functions
|       |-- joystick.py - pyGame joystick input functions
|       |-- network.py - FlightGear network interface functions
|       |-- rcam_terrain.xml - definition file for FlightGear to send terrain height back to Python
```
(*) from this video: [ISA](https://youtu.be/TZJ3B89REHw)  
(**) from this video: [Turbofan Deck](https://youtu.be/95Gy2wg3olE)  

## Program Options/Settings:  
### Aircraft
The aircraft parameters are defined in file *rcam_parameters.json*  
You are encouraged to create other aircraft models and add additional files.  

In **main.py**  
Select the trim condition, if on ground or in air: TRIM_ON_GROUND  
Intial altitude: INIT_ALT_FT  
Trim speed: V_TRIM_MPS  
Postion: INIT_LATLON_DEG  
Flaps position: FLAPS_INIT  
Gear position: INIT_GEAR  
Flight path angle: GAMMA_TRIM_RAD  
Heading: INIT_HDG_DEG  
Wind: WIND_NED_MPS  
Wind Std Dev: WIND_STDDEV_MPS  
Engine Deck file: (set in propulsion.py)  
### Simulation
Total simulation time: SIM_TOTAL_TIME_S  
Main simulation loop frame rate: SIM_LOOP_HZ  
Data rate output to FlightGear: FG_OUTPUT_LOOP_HZ  
Engine deck calculation rate: DECK_LOOP_HZ  
Simulator/FG visual Z offset: SIM_VISUAL_OFFSET
Use FlightGear as terrain database: USE_FG_AS_TERRAIN_DB (if set to false, it will use SRTM data instead)  
Data logging rate: DATA_LOGGING_HZ  
Selected engine parameters to be output/logged: ENG_LOG_PARAMETERS

Results file (CSV): RESULTS_FILE  
Select time interval to save to disk: LOG2DISK_INTERVAL_S  

### FlightGear comms
Main instance for visual: UDP_IP1  
Second instance for visual: UDP_IP2  

Aiming for performance, the program runs the integration loop at a target pf 400Hz, adjusting the integration steps to the available computing cycles.  
It uses Numba to speed up the main functions involved in the integration loop.  

Output is sent to FlightGear (FG), over UDP, at a reduced frame rate (set to 60Hz).  
The FG interface uses a stripped down version of the class implemented by Andrew Tridgel [fgDFM](https://github.com/ArduPilot/pymavlink/blob/master/fgDFM.py). It was simplified and a few additional parameters were added. It is compliant with version 24 of FlightGear protocol.  

Currently, the first UDP address is set to the local machine, with a second UDP address available for an extra screen/instance of FG.

If a known joystick is detected, then inputs come from it  
Currently, only mappings for the Logitech Extreme 3D joystick is implemented. You will need to add your own to the JSON config file (follow similar structure).  
Otherwise, offline simulation is run.

Before running the code/setup:  
1 - for FG terrain database comms, place a copy of psim/io/rcam_terrain.xml in [your FlightGear root directory]/fgdata/Protocol/  
(in my case, this is /home/[USERNAME]/FlightGear/fgdata/Protocol) 
2 - for SRTM terrain database, place a copy of psim/io/N47E011.hgt in [your home directory]/.cache/srtm/  
(in my case, this is /home/[USERNAME]/.cache/srtm)  
3 - if available, plug joystick in  

To run:  
1 - from a terminal, start the FlightGear instance that will render visuals and inform the terrain height. For example, in my setup I have FlightGear installed into its own directory. After cd'ing into it, I run this:  
./flightgear-2024.1.3-linux-amd64.AppImage --airport=LOWI  --aircraft=757-200-RB211 --native-fdm=socket,in,60,,5500,udp --fdm=null --enable-hud --in-air --fog-disable --shading-smooth --texture-filtering=4 --timeofday=morning --altitude=2500 --prop:/sim/hud/path[1]=Huds/NTPS.xml --generic=socket,out,5,127.0.0.1,5502,udp,rcam_terrain --prop:/controls/flight/elevator-trim=0

2 - from its own terminal, run main.py  


____________________________________________________________________________________________
Here is the simulation flowchart:

```mermaid
graph TD
    %% Global Styles
    %% Main Loop (Blue): Shows the sequential execution of every frame (Sensors -> Inputs -> Logic -> Actuators -> Physics -> Output).
    classDef process fill:#9B4501,stroke:#01579b,stroke-width:2px;
    classDef decision fill:#2D68FB,stroke:#fbc02d,stroke-width:2px;
    %% input/output stuff (magenta)
    classDef io fill:#7D2E79,stroke:#2e7d32,stroke-width:2px;
    %% Parallel Workers (Purple/Dotted): Shows that Networking and Engine calculations happen asynchronously to the main loop.
    classDef thread fill:#46A21F,stroke:#7b1fa2,stroke-width:2px,stroke-dasharray: 5 5;
    %% Physics Core (Red): This highlights the mathematically heavy parts (Trimming, RCAM Model, Ground Forces).
    classDef physics fill:#1d9191,stroke:#c62828,stroke-width:2px;

    Start([Start PSim.py]) --> LoadParams[Load rcam_parameters.json]
    LoadParams --> InitJoy[Init Pygame & Joystick]
    InitJoy --> CheckMode{Joystick Present?}
    
    %% Initialization Phase
    subgraph Init [Initialization Phase]
        CheckMode -- Yes --> OnlineMode[OFFLINE False]
        CheckMode -- No --> OfflineMode[OFFLINE True]
        OnlineMode --> StartThreads[Start Network & Engine Threads]
        
        subgraph Workers [Background Workers]
            direction TB
            NetTX(Network TX Thread<br/>Python UDP -> FG FDM)
            NetRX(Terrain RX Thread<br/>FG Terrain -> Python UDP)
            EngProc(Engine Process<br/>Multiprocessing)
        end
        
        StartThreads -.- NetTX
        StartThreads -.- NetRX
        StartThreads -.- EngProc
        
        OnlineMode --> TrimCalc
        OfflineMode --> TrimCalc
        
        TrimCalc[Trim Aircraft<br/>Minimize Cost Function]:::physics
        TrimCalc --> SetIC[Set Initial Conditions & Integrators]
    end

    SetIC --> SimLoop{Time < Max?}

    %% Main Simulation Loop
    subgraph MainLoop [400Hz Simulation Loop]
        SimLoop -- Yes --> DesatThrottle[1. De-saturate Throttles]
        
        DesatThrottle --> ReadInputs[2. Read Inputs]
        
        ReadInputs --> GetJoy[* Get Joystick Axes/Buttons]:::io
        GetJoy --> Logic[3. Logic & Saturation]
        
        Logic --> AirGnd[* Check Air/Ground State]
        AirGnd --> Spoilers[* Spoilers & Auto-Brake]
        Spoilers --> Actuators[4. Actuator Dynamics]
        
        Actuators --> Lag[* Apply 1st Order Lag<br/>Surfaces & Brakes]
        
        %% GETTING THRUST (Reading from Queue)
        Lag --> EngRead[5. Get Async Engine Thrust]
        EngProc -.-> EngRead
        
        EngRead --> Physics[6. Physics Integration]
        
        subgraph PhysicsEngine [Physics Core]
            Physics --> SetParams[Set integrator params<br/>U, rho, h, gnd]
            SetParams --> IntegrateAC[Integrate Aircraft<br/>RK45 / dopri5]
            
            IntegrateAC --> CallModel[[RCAM_model]]:::physics
            CallModel --> GroundForce[Calc Ground Forces<br/>Spring/Damper/Friction]:::physics
            GroundForce --> Aero[Calc Aero Forces/Moments]:::physics
            Aero --> Equations[6-DOF Equations of Motion]:::physics
            
            Equations --> NewState[New State X]
        end
        

        NewState --> Nav[7. Integrate Navigation<br/>Lat/Lon/Alt]
        
        Nav --> GetTerrain[8. Get Ground Altitude]:::io
        NetRX -.-> GetTerrain
        GetTerrain --> CalcAGL[9. Calculate AGL & Density]
        CalcAGL --> Logging[10. Logging & Output]
        Logging --> Observe[[* RCAM_observe]]:::physics
        Observe --> DataCol[* Append to Data Collector]
        
        %% TRIGGERS
        DataCol --> CheckFrame{60Hz FG Trigger?}
        
        CheckFrame -- Yes --> SendFG[Send UDP Packet to FlightGear]:::io
        SendFG -.-> NetTX
        CheckFrame -- No --> CheckEngTrigger
        
        %% NEW ENGINE TRIGGER LOGIC
        CheckEngTrigger{10Hz Engine Trigger?}
        SendFG --> CheckEngTrigger
        
        CheckEngTrigger -- Yes --> DispatchEng["Dispatch New Job<br/>(Alt, Mach, TLA)"]:::io
        DispatchEng -.-> EngProc
        CheckEngTrigger -- No --> RateLimit
        DispatchEng --> RateLimit
        
        RateLimit[Wait/Rate Limiter] --> SimLoop
    end

    %% Termination
    SimLoop -- No --> Cleanup[Stop Threads & Processes]
    Cleanup --> SaveCSV[Save test_data.csv]:::io
    SaveCSV --> Plot[Generate & Show Plots]:::process
    Plot --> End([End])

    %% Styling mappings
    class DesatThrottle,LoadParams,InitJoy,SetIC,ReadSensors,CalcAGL,ReadInputs,Logic,AirGnd,Spoilers,Actuators,Lag,EngRead,Physics,SetParams,IntegrateAC,NewState,Nav,Logging,DataCol,RateLimit,Cleanup process;
    class CheckMode,SimLoop,CheckFrame,CheckEngTrigger decision;
    class GetTerrain,GetJoy,SendFG,DispatchEng,SaveCSV io;
    class NetTX,NetRX,EngProc thread;
    class TrimCalc,CallModel,GroundForce,Aero,Equations,Observe physics;
```
