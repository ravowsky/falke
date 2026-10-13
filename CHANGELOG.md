# Changelog

Unreleased
===
- Add THAW / reTHAWed compatibility (`thaw.exe`, `rethawed.exe`)
  - Freecam keyframing and camera paths, including FOV
  - Replays record the skater, the camera, cars and pedestrians
  - Skater replays play back in freecam
  - Crashes and panics are written to falke.log with the game code that was running
- Replay window is available outside freecam, so a recording can be named and started while skating
- Playback speed now also slows down the game world (THAW): traffic, pedestrians and particles follow the replay at 0.1x
- Smooth slow-motion skater playback (THAW)
  - Bones are blended in world space, so turns the game undoes in one frame (switch/fakie landings) no longer swing the body around
  - Animation cuts and wallride/landing transitions are eased out instead of snapping
  - Joints stay connected while blending, no more stretched arms or exploding hands
- Cars, pedestrians and other animated objects are synced to replay playback (THAW)
  - The 24 pedestrians and 24 other animated objects nearest to the skater are recorded
  - Cars that were not recorded are hidden during playback
  - Only syncs for replays recorded in the current game session
- Experimental: sync moving objects without a skeleton, such as cranes and crates (THAW), switched on in the Replay window
- The game world stands still while a replay is paused in freecam (THAW)
- Generic replay interpolation no longer blends across 180° flips and animation cuts (shows the nearer frame instead)
- Replay format version 4: cars and pedestrians are stored at half precision, about 45% smaller files; version 3 replays still load
- Faster replay loading
- Game backends can provide their own skater blending and world time scale

0.6.0
===
- Add THPS4(Better4) compatibility for keyframing and replays
- Use egui software cursor instead of windows cursor
- Have replay functionality be toggled per game
- POV Mode now only shows when replay is available and loaded
- Replay widget now only shows when replay is available
- Unselects keyframe when not on keyframe
- Fix auto focus on duration input
- Restore FOV when going in and out of freecam
- Fix delay in applying playback info only after scrubbing the timeline
- Loading a replay sets cam mode to POV
- POV mode does not allow keyframes
- Add red indicator if camera moves off keyframe
- UI Improvements

0.5.0
===
- Enabled keyframes to be edited directly
- Allow manual edit of rotation
- Add basic replay functionality
  - Records every skater's pose and the camera to a ".flk" file
  - Load a recorded replay and play it back along the timeline
- Add camera perspective toggle for Freecam and POV
- Add Orbit camera mode
  - Orbits the replay's skater, or a fixed point when no replay is loaded
  - Drag: Orbit
  - Shift + Drag: Pan
  - Wheel: Zoom
- Add new Hotkeys
  - F2: Start/Stop Recording
- Restructured UI for more space
- Add zooming and scroll functionality for timeline
- Add adjustable timeline duration
- Add "Fit" button to zoom the timeline back out to the full duration
- Fixed camera speed jumping between unevenly spaced keyframes
- Add groundwork for supporting more games

0.4.0
===
- Rewrite with user interface

0.3.1
===
- Fixed falke to work with newest THUGPro Update

0.3.0
===
- Add FOV Interpolation
- Add new Hotkeys
  - F3: Decrease FOV
  - F4: Increase FOV
  - F10: Toggle Camera Preview Path
  - F11: Delete Last Camera Snapshot
- Camera Path related hotkeys now only work in Freecam Mode

0.2.0
===
- Rewrite
- Add Cubic Bezier Interpolation
- Rework Hotkeys
- Rename "thook" to "falke"
- Change log path to "{GameFolder}/falke.log" 
  instead of creating multiple log files in log folder
- DLL no longer needs to be in folder of Game Executable
- Add Debugline of Camera Path (disabled once playing)
- Add Configuration File for Executable Name
- Add possibility to change playback speed

0.1.0
===
- Initial Private Release