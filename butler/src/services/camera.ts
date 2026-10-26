import { BunServices } from "@effect/platform-bun";
import { Console, Context, Effect, Layer, Schema, Semaphore, Stream } from "effect";
import { ChildProcess, ChildProcessSpawner } from "effect/process";

export type Image = File;
export type StereoPair = { readonly left: Image; readonly right: Image };

export class Camera extends Context.Service<Camera, {
  // Capture paired left/right views; USB cameras are not hardware-synchronized.
  readonly capture: Effect.Effect<StereoPair, Error>;
}>()("butler/Camera") {}

export const MockCamera = Layer.succeed(Camera, {
  capture: Effect.gen(function*() {
    yield* Console.log("[camera:mock] capture — started");
    yield* Effect.sleep("50 millis");
    yield* Console.log("[camera:mock] capture — finished");
    return {
      left: new File([], "left.png", { type: "image/png" }),
      right: new File([], "right.png", { type: "image/png" }),
    };
  }),
});

// Python OpenCV avoids native Node/V8 bindings that are not portable to Bun.
const cameraOptions = { python: "python3", left: 1, right: 2, width: 1280, height: 720, warmupFrames: 5 };
const PngPair = Schema.fromJsonString(Schema.Struct({
  left: Schema.Uint8ArrayFromBase64, right: Schema.Uint8ArrayFromBase64,
}));

const pngFile = (data: Uint8Array, side: string) => new File([
  new Uint8Array(data),
], `${side}.png`, { type: "image/png" });

// Open per capture so there are no idle camera buffers or long-lived workers.
// Grab both cameras before decoding either frame to minimize software skew.
const captureScript = String.raw`
import base64
import json
import sys
import cv2

left_index, right_index, width, height, warmup = map(int, sys.argv[1:])
if left_index == right_index:
    raise ValueError("Left and right camera indices must differ")

cameras = []
try:
    for index in (left_index, right_index):
        camera = cv2.VideoCapture(index)
        cameras.append(camera)
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open USB camera {index}")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    for _ in range(warmup + 1):
        grabbed = [camera.grab() for camera in cameras]
        if not all(grabbed):
            raise RuntimeError("Failed to grab a stereo pair")
        frames = [camera.retrieve() for camera in cameras]
        if any(not ok or frame is None for ok, frame in frames):
            raise RuntimeError("Failed to retrieve a stereo pair")

    if frames[0][1].shape != frames[1][1].shape:
        raise RuntimeError("Stereo cameras returned different image dimensions")

    pair = {}
    for side, (_, frame) in zip(("left", "right"), frames):
        ok, png = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError(f"Failed to encode {side} image")
        pair[side] = base64.b64encode(png).decode("ascii")
    print(json.dumps(pair))
finally:
    for camera in cameras:
        camera.release()
`;

export const LiveCamera = Layer.effect(Camera, Effect.gen(function*() {
  const spawner = yield* ChildProcessSpawner.ChildProcessSpawner;
  const lock = yield* Semaphore.make(1);
  return Camera.of({
    capture: Effect.gen(function*() {
      yield* Console.log("[camera:live] capture — started");
      const { python, left, right, width, height, warmupFrames } = cameraOptions;
      const process = yield* spawner.spawn(ChildProcess.make(python, [
        "-c", captureScript, ...[left, right, width, height, warmupFrames].map(String),
      ], { stderr: "inherit", killSignal: "SIGKILL" }));
      const [stdout, exitCode] = yield* Effect.all([
        process.stdout.pipe(Stream.decodeText(), Stream.mkString), process.exitCode,
      ], { concurrency: 2 });
      if (exitCode !== 0) return yield* Effect.fail(new Error(`Stereo capture failed (exit ${exitCode})`));
      const pair = yield* Schema.decodeUnknownEffect(PngPair)(stdout);
      yield* Console.log("[camera:live] capture — finished");
      return { left: pngFile(pair.left, "left"), right: pngFile(pair.right, "right") };
    }).pipe(Effect.scoped, Effect.timeout("15 seconds"), lock.withPermit),
  });
})).pipe(Layer.provide(BunServices.layer));
