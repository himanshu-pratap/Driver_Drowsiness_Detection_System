# Driver_Drowsiness_Detection_System

## Context
Driving when you are sleepy & exhausted? Well, you're as much of a safety hazard as a drunk driver. “Drowsy driving” occurs when a person who is operating a motor vehicle is too tired to remain alert. The effect of drowsiness is similar to alcohol; it will make your driver inputs(steering, acceleration, and braking) poorer, destroy your reaction time and blur your thought processes.

Now it is high time we take control and help in reducing these kinds of accidents that pose a severe risk of life.

## Solution
In order to mitigate the drowsy-driving accidents ,we are using a set of multiple technologies such as: OpenCV + Tkinter in Python3. This interface system will consist of a camera which will turn on whenever the driver starts running the main engine and then it will check two major aspects of drowsiness:
* Yawning- checked through facial landmark detection.
* Eye closing- analyzed through eye-aspect ratio.

## Process
Whenever the driver yawns or closes his eyes for more than threshold time then the alarm goes on to signal the driver as well as the respective activity count is increased by one. On the closing of software the totals of each activity count will be displayed on screen using tkinter.

## How to run?
* Run python script using python yawn_detector.py
* Wait for script to start and then test it using closing eyes or yawning. Use keystroke q to close window and see this run analytics.
* On pressing 'q' the tkinter will launch a dialog box that will represent the number of eyes closed and number of yawn taken.

## Conclusion
This model, if implemented correctly on a large scale, can prevent losses in road accidents due to drowsy driving and save thousands of lives. The system can be installed in different vehicles whether those are buses , trucks, cars or even in the railways.
