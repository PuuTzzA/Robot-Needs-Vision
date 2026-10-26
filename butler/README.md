# Robot butler

Effect/TypeScript controller for Stage 2 of [the project](task.md).

```sh
bun install
bun run typecheck
bun run start
```

**Check the layers in `src/index.ts` before running.** The current selection uses
live navigation and OpenAI, with mock transport, camera and vision. It makes
billed API requests but does not drive the robot. **Mock captures are empty PNG
`File`s, not valid image data:** select `MockLanguageModel` for offline runs or
`LiveCamera` for real visual tasks. Set `OPENAI_API_KEY` for OpenAI.

`rclnodejs` is loaded only when `LiveTransport` is provided; mock runs need no
native ROS dependencies.

## Layout

- `src/domain.ts` — destinations, coordinate schema and mapped poses.
- `src/index.ts` — choose each service's mock or live layer.
- `src/program.ts` — read the board, plan, inspect, answer and return home.
- `src/services/` — one file per service, mock first, then live implementation:
  - `camera.ts` — capture a `{ left, right }` stereo pair.
  - `vision.ts` — both layers currently return mock detections. The live generator
    includes a commented multipart HTTP request sending both files to `/detect`.
  - `navigation.ts` — return ordered `[destination, coordinate]` pairs. Basic live
    navigation preserves input order; advanced route optimization is unfinished.
  - `transport.ts` — send a Nav2 goal and wait for arrival or failure.
  - `language-model.ts` — board interpretation, location analysis and final answers.

Parameters stay inline in the relevant service or controller. Configure all
seven poses in `domain.ts` for the current map before live driving; the supplied
zeros are placeholders. Coordinates are metres, headings are degrees, and
`home` must face the task board. Reload poses when the level's scene/map changes.

## Stereo camera

Live capture uses Python OpenCV through Effect's child-process API:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install opencv-python
```

Set `cameraOptions.python` in `src/services/camera.ts` to that interpreter
(prefer an absolute path). Adjust left/right USB indices (defaults **1 and 2**),
requested resolution and warm-up frames there. Camera indices can change after
reconnection; confirm their order against your calibration.

Each capture opens both cameras, warms up, grabs both before retrieving either,
and returns lossless PNG `File`s. Cameras are released afterward; reopening adds
startup overhead but avoids idle buffering. Captures are serialized with a
15-second timeout; timeout/interruption kills the subprocess. Python errors go
to stderr and failed captures fail the service call.

Independent USB cameras are **not hardware-synchronized**. Check the actual
resolution (backends can ignore requests), match camera settings and calibrate
for that setup. Rectification and depth estimation belong in vision.

## Execution

The LLM reads the board's left image and returns destinations or `"all"` for a
search/inventory. Driving and capture are sequential; up to three location
analyses run while driving continues. Vision gets both views; the LLM only gets
the left image plus detections/heights. Final synthesis overlaps the return home,
and output waits for both to finish. Raw evidence accompanies each analysis.

Mocks return fixed data (empty image files for camera) without tracking levels
or positions. Every mock call
logs start/finish around a short delay: camera/navigation 50 ms, vision 100 ms,
LLM 400 ms, transport 700 ms. The fixed two-point plan demonstrates overlap.

Live transport requires sourced ROS 2, a working `rclnodejs` native build, active
Nav2, localization and `nav2_msgs`. Its scoped
ROS context/node are released on shutdown. Goals fail if unavailable, rejected
or unsuccessful. **There is no automatic goal cancellation:** a timeout or
interruption does not stop the robot. Use Nav2 or the physical stop when needed.

The controller allows 570 seconds of work and a 30-second return attempt per
level, not a shared 10-minute budget across all five tasks. Level transitions
are not automated. Live camera/robot operation has not been hardware-verified.
