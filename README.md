# falke
A tool for movie making enhancements for Neversofts Tony Hawk's Pro Skater series.

## Features
- Enhances the Free Cam of Tony Hawk's Pro Skater games to play interpolated Camera Paths between placeable snapshots (supports Transform, Rotation and FOV).
Also adds basic replay functionality.

## Installation
I recommend using the "Ultimate ASI Loader" to inject the falke.dll into the necessary process.
Installation might differ for games.

### Simple way (bundled)
Just download the release zip and put it inside your `Game` folder.

### From source (code, not me).
You have to build the .dll with cargo. You must inject it to the process (by using Ultimate ASI Loader f.e.).
For THPS4 you also need my d3d8to9 fork (https://github.com/sourcethps/d3d8to9) as the game is in Direct3D 8.
Put that as d3d8.dll and ASI Loader as dinput8.dll

## Troubleshooting
If something goes wrong, you need to provide a falke.log file.

### Using Reshade with this
If you use Ultimate ASI Loader, you can just put reshade as reshade.asi inside the plugins folder.
For THPS4 you need to use d3d8to9 as Reshade only works with Direct3D 9.

## How-To
The overlay should show once you move the camera in free cam mode.

You need at least two keyframes for a playable path. They are marked in the timeline and can be dragged.

Toggle the "Playback mode" to be able to enter freecam.

For playing the path hit the **SPACE** Key.

Press **F1** to toggle the UI. This is useful when recording. I advise to put the playtime marker a couple of seconds before the first keyframe, press play (Space) and toggle the UI (F1) when you want to record.

## Known Issues
- If the overlay doesn't show, try restarting the game
- When in replay, the stance and board state matters.
Your board is missing? Get out of freecam, get on the board, go into the replay again.
Your tricks spin wrong? Get out of freecam, change your stance, go into the replay again.
