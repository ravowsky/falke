# falke
A tool for movie making enhancements for Neversofts Tony Hawk's Pro Skater series.

Supported games:
- THUGPro (THUG2 mod)
- Better4 (THPS4 mod)


## Features
- Enhances the Free Cam to play interpolated camera paths between placeable keyframes
  (position, rotation and FOV).
- Three camera modes to shoot in, fly it yourself, ride the replay's own camera, or
  orbit the skater.
- Basic replay recording and playback.

There is no video export. You frame the shot with falke and capture it with OBS or
whatever else you record with.

## Installation
I recommend using the "Ultimate ASI Loader"
(https://github.com/ThirteenAG/Ultimate-ASI-Loader) to inject falke into the game.
falke is loaded as a plugin, so the built `falke.dll` is renamed to `falke.asi` and
put in a `plugins` folder.

THPS4 is a Direct3D 8 game and falke's overlay is Direct3D 9, so it needs my d3d8to9
fork (https://github.com/sourcethps/d3d8to9) to run at all.

### Simple way (bundled)
Download the release zip and put its contents inside your game folder.

### From source (code, not me)
You need a Rust toolchain with the 32-bit Windows target and the MSVC build tools:

```
rustup target add i686-pc-windows-msvc
cargo build --release
```

`.cargo/config` already pins the target, so the output lands in
`target/i686-pc-windows-msvc/release/falke.dll`. Rename it to `falke.asi` and put it
in the game's `plugins` folder.

## Workflow
The overlay shows once you move the camera in free cam mode. **F1** toggles it, so you can
hide it right before you start capturing.

1. **Record a run.** Type a name in the Replay window and hit **F2** (or the record
   button). F2 does nothing until a name is typed. Press **F2** again to stop. The
   replay is written to `Replays/<name>.flk` next to the game.
2. **Load it.** Type the same name into the Replay window's Load box and press Load.
   There is no file browser, so the name is how you find it again.
   Loading a replay clears any camera path you had authored and switches you to POV.
3. **Author the shot.** Tick **Playback mode**. That is what lets the timeline
   drive the camera, in any mode. Pick a camera mode, scrub to a time, frame the camera and
   hit **➕ Add**. You need at least two keyframes at different times for a playable
   path. Markers can be dragged. **Delete** removes the selected one.
4. **Play it.** **SPACE** plays and pauses.
5. **Capture it.** Park the playhead a couple of seconds before the first keyframe,
   press Space, then F1 to hide the UI, and record your screen.

You cannot record a replay and author a camera path at the same time. falke leaves the
camera alone while recording, so the camera you shoot with is the one you fly
afterwards.

### Camera modes
They are mutually exclusive, and each one ignores what another uses:

| Mode | What drives the camera | Ignores |
|---|---|---|
| Freecam | your keyframe path | the replay's recorded camera |
| POV | the camera the replay was shot with | your keyframes (needs a replay loaded) |
| Orbit | drag to circle the skater, or the path, whichever got the last input | nothing |

So "my keyframes do nothing" usually means you are in POV, and "my replay's camera was
never recorded" usually means you are in Freecam.

In orbit mode the pivot is the skater whenever a replay is supplying one, and a point in
the level otherwise. Drag to orbit, shift+drag to pan, wheel to zoom.

### How a replay works
A `.flk` file stores the skater's bone poses and the camera, frame by frame. It does
not store *your skater*. Playback poses the live skater and draws that, so your current
board and stance are what ends up on screen. That is why the two issues below happen,
and why getting out of freecam and fixing it on the live skater is the fix.

## Troubleshooting
If something goes wrong, you need to provide a `falke.log`. It is written next to the
game executable.

### Using Reshade with this
If you use Ultimate ASI Loader, you can just put reshade as reshade.asi inside the
plugins folder. For THPS4 you need d3d8to9 as Reshade only works with Direct3D 9.

## Known Issues
- If the overlay doesn't show, try restarting the game.
- When in replay, the stance and board state matters.
  Your board is missing? Get out of freecam, get on the board, go into the replay again.
  Your tricks spin wrong? Get out of freecam, change your stance, go into the replay again.
- During playback the live skater is still standing where you left it and the game keeps
  drawing it, so panning the camera past that spot brings a second skater into shot.
- One skater per replay.
