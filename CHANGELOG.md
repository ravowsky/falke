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
  - Wallrides: the snap onto and off the wall is eased out in the skater's own space, so the fade turns with the skater (it turns about 17° a frame on the wall) instead of twisting against it
  - Turning skaters (wallrides, spins) are blended relative to the skater, so the body swings round with the turn instead of cutting across it
  - The skater keeps its speed into a cut and curves into the new motion, instead of stalling at the cut
  - Motion and turns follow a smooth curve through the recorded frames instead of changing direction at every frame (visible as a judder in slow motion)
- Landing fakie into a manual no longer spins the skater all the way round in playback: the game shows the landing frame with the skater turned the wrong way round for one frame, which is now turned back. Turns of more than 60° in one frame (flips, snaps) no longer count as a speed to carry on with when easing out an animation cut
- An animation cut right before a teleport (respawn, big snap) no longer jumps at the frame before it
- Cars, pedestrians and other animated objects are synced to replay playback (THAW)
  - The 24 pedestrians and 24 other animated objects nearest to the skater are recorded
  - Cars that were not recorded are hidden during playback
  - Syncs after restarting the game too: objects are recorded by their id (the checksum of their name, the same every time the level loads) and played on the live object with that id. A recorded object with no match (a pedestrian spawned differently, or a replay recorded before this) is played on the nearest free live object of the same kind and skeleton, which keeps it while the replay plays; it moves as recorded but may look different. Replays recorded before this still sync fully in the game session they were recorded in
- Experimental: sync moving objects without a skeleton, such as cranes and crates (THAW), switched on in the Replay window
- The game world stands still while a replay is paused in freecam (THAW)
- Generic replay interpolation no longer blends across 180° flips and animation cuts (shows the nearer frame instead)
- Replay format version 4: cars and pedestrians are stored at half precision, about 45% smaller files; version 3 replays still load
- Replay format version 5: every frame stores when it was captured, so frames the game took longer for (hitches) play back at their real pace instead of as a lurch; frames identical to the one before are not stored. Versions 3 and 4 still load
- Replay format version 6: frames are compressed in one-second blocks (each number stored as its difference from what the two frames before predict, then deflated), on a background thread while recording. Replays are about a third of their old size with nothing lost; the skater's unused rotation slots are no longer stored. A recording cut short (game crash) still loads up to its last second. Versions 3 to 5 still load
- Release zip has a `rethawed` folder (falke.asi and the ASI loader)
- Overlay no longer crashes the game with "unable to draw indexed prims" (D3DERR_INVALIDCALL) when the game ends its frame with a depth buffer of another size, extra render targets or instancing bound, or when the window is larger than the back buffer (some resolutions / DPI scaling). A draw that still fails skips that frame's UI instead of crashing. egui-d3d9 is now vendored in `egui-d3d9/` (changes in `egui-d3d9/FALKE-PATCHES.md`)
- Frame-locked video capture (Capture window): the replay, camera path and game world move on by exactly one output frame per drawn frame, at any frame rate (e.g. 500 fps) and speed, so nothing is dropped or doubled
  - ProRes 4444 / ProRes 422 HQ / H.264 through ffmpeg (`ffmpeg.exe` next to the game), or PNG / TGA sequences without ffmpeg
  - Passes (THAW): Image, Depth (16-bit PNG, linear distance, read from the depth buffer swapped for a readable INTZ one while capturing), World (redrawn without the skater, world held still), Skater (the image with alpha where it differs from World)
  - Output in `Captures\<name>\` next to the game
  - Image from Screen: Image, World and Skater read after ReShade and its add-ons (DLSS, RenoDX), from the DirectX 11 swap chain dgVoodoo presents through. falke registers as a ReShade add-on and reads the frame at `reshade_present`; each frame is held for a few "settle" frames so DLSS / temporal effects have caught up; the overlay hides while capturing (End stops); HDR10 / scRGB are converted to SDR
  - View in game: any pass shown live in place of the normal image (Depth as grey, World without the skater, Skater over a checkerboard) to check it before capturing
- Bone camera mode (THAW): the camera rides one of the replayed skater's bones (board cam, head cam), with offset, turn, horizon levelling and stabilizing; bones can be picked by clicking their numbers on screen
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