# Changelog

0.7.0
===
- Fixing inconsistencies in UX
- Camera dirty state is now derived from difference to keyframe data
  - Show what is changed and provide Apply and Reset Buttons
- Separated "Play" and "Fly" mode
- "Playback mode" is now the better fitting "Preview on scrub" option
- Improved Replay loading UI

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
