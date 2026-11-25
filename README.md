# falke
A tool for movie making enhancements for Neversofts Tony Hawk's Pro Skater series.

Supported games:
- THUGPro (THUG2 mod)
- Better4 (THPS4 mod)
- THAW / reTHAWed (Tony Hawk's American Wasteland)


## Features
- Enhances the Free Cam to play interpolated camera paths between placeable keyframes
  (position, rotation and FOV).
- Four camera modes to shoot in: fly it yourself, ride the replay's own camera, orbit the
  skater, or fix the camera to one of the skater's bones.
- Basic replay recording and playback.
- Frame-locked video capture of the replay (THAW / reTHAWed): any frame rate, no dropped
  frames, with depth, world-only and skater-only passes.

## Installation
I recommend using the "Ultimate ASI Loader"
(https://github.com/ThirteenAG/Ultimate-ASI-Loader) to inject falke into the game.
falke is loaded as a plugin, so the built `falke.dll` is renamed to `falke.asi` and
put in a `plugins` folder.

THPS4 is a Direct3D 8 game and falke's overlay is Direct3D 9, so it needs my d3d8to9
fork (https://github.com/sourcethps/d3d8to9) to run at all.

### Simple way (bundled)
Download the release zip and put the contents of your game's folder (`thugpro`,
`better4` or `rethawed`) inside your game folder.

For reTHAWed that is the folder with `reTHAWed.exe` (`thaw\Game`). If you already use an
ASI loader there (a `dinput8.dll` from Ultimate ASI Loader, e.g. for ReShade or other
plugins), keep yours and only copy `plugins\falke.asi`.

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
5. **Capture it.** Open the **Capture** window, pick a name, format, frame rate and range,
   and press **🎬 Capture** (see [Capture](#capture)). Or park the playhead a couple of
   seconds before the first keyframe, press Space, then F1 to hide the UI, and record your
   screen.

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
| Bone | one of the replayed skater's bones, with an offset (THAW) | your keyframes (needs a replay loaded) |

So "my keyframes do nothing" usually means you are in POV, and "my replay's camera was
never recorded" usually means you are in Freecam.

In orbit mode the pivot is the skater whenever a replay is supplying one, and a point in
the level otherwise. Drag to orbit, shift+drag to pan, wheel to zoom.

### Bone camera
Pick **Bone** as the camera mode and the camera rides one of the replayed skater's bones:
a board cam, a head cam, a hand cam. With **Show** ticked, every bone's number is drawn
over the game; click one to pick it, or step through them with ◀ ▶. **Offset** moves the
camera along the bone's own axes, **Turn** turns it (yaw, pitch, roll).
**Turn with the bone** off keeps only the bone's position (the camera looks the way Turn
says, in the world). **Keep the horizon level** drops any roll. **Stabilize** smooths the
bone's motion over that many seconds of the replay (also in captures, it is worked out
from the replay, not from the frame rate).

### Capture
The **Capture** window (freecam, overlay on) renders the replay to files, one frame at a
time: falke moves the replay, the camera and the game world on by exactly one output frame
for every frame the game draws, however long saving takes. So 500 fps is as smooth as
30 fps, and nothing is dropped or doubled. The overlay is not in the captured frames.

- **Format:** ProRes 4444, ProRes 422 HQ or H.264 need `ffmpeg.exe` next to the game's exe
  (or on the PATH); PNG and TGA sequences do not. Everything goes to
  `Captures\<name>\` next to the game.
- **Frame rate** and **Speed:** speed 0.1 at 60 fps gives ten times slow motion at
  60 fps; speed 1 at 500 fps gives footage to slow down later.
- **From / to:** the replay time range (**All**: the whole replay and path).
- **Passes** (THAW / reTHAWed):
  - **Image**: what the game draws.
  - **Depth**: 16-bit PNGs, black at the camera, white at the distance set under
    "Depth white at" (game units). For depth of field or fog in an editor. Needs a graphics
    driver with readable depth buffers (INTZ), and no anti-aliasing on the game's depth
    buffer.
  - **World**: every frame drawn a second time with the skater hidden (the world held
    still in between).
  - **Skater**: the image with an alpha channel that keeps only where Image and World
    differ: the skater, and what it changes (its shadow, reflections). ProRes 4444 or PNG
    keep the alpha; with H.264 this pass is written as PNGs.

**View in game** shows a pass live in the game instead of the normal image, so you can
check it while you frame the shot: **Depth** as grey (set its range with "Depth white at"),
**World** without the skater, **Skater** as the skater over a checkerboard (it compares
every other frame, so pause playback to check the threshold). World and Skater need a
replay playing or paused in the freecam. Starting a capture switches the view back to Image.

**Image from** picks where the Image, World and Skater passes are read:
- **Game**: the game's own frame (Direct3D 9), before anything further down the chain.
- **Screen**: what reaches the screen, after ReShade and its add-ons (DLSS, RenoDX, ...).
  Needs ReShade with add-on support running on dgVoodoo's DirectX 11 output (ReShade as
  `dxgi.dll` next to the game); falke registers itself with ReShade as an add-on (it shows
  up in ReShade's Add-ons tab as falke) and reads the frame just before it is presented.
  - Effects that run behind the game (DLSS, temporal AA, ...) show a frame or more late, so
    every output frame is drawn again, the world held still, for **Settle** frames before
    it is saved. 2 is enough with the DLSS 5 feeder; raise it if frames come out one step
    late (a stutter where two frames are the same), lower it to capture faster.
  - falke's overlay would be in the frames, so it hides while capturing. Press **End** to
    stop early. Close ReShade's own overlay too, and turn off its clock / FPS display: they
    are drawn into the same image.
  - HDR (HDR10 or scRGB swap chains) is brought down to SDR with 203 nits as white.
  - The Skater matte compares two renders of the same moment. Neural rendering and
    temporal effects do not draw them exactly alike, so the matte gets noisier: raise
    "Skater threshold", or capture the matte with **Game**.
  - Depth is always read from the game.

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
- Capture passes other than Image, and the bone camera, are for THAW / reTHAWed only.
- The Depth pass reads the depth buffer the game draws the level with. Anything the game
  draws with its depth buffer cleared afterwards (the HUD) is not in it.
