import { Console, Context, Effect, Layer } from "effect";
import type { StereoPair } from "./camera.ts";

export class Vision extends Context.Service<Vision, {
  // Raw detector output and depth-derived heights are passed straight to the LLM.
  readonly detectWithHeights: (images: StereoPair) => Effect.Effect<unknown, Error>;
}>()("butler/Vision") {}

export const MockVision = Layer.succeed(Vision, {
  detectWithHeights: Effect.fnUntraced(function*() {
    yield* Console.log("[vision:mock] detectWithHeights — started");
    yield* Effect.sleep("100 millis");
    yield* Console.log("[vision:mock] detectWithHeights — finished");
    return "Mock detections; no measured heights.";
  }),
});

export const LiveVision = Layer.succeed(Vision, {
  detectWithHeights: Effect.fnUntraced(function*(images: StereoPair) {
    yield* Console.log(`[vision:live] detectWithHeights — started`);

    // Replace the mock delay/result below with this request when the backend is ready.
    // const body = new FormData();
    // body.set("left", images.left);
    // body.set("right", images.right);
    // const result = yield* Effect.tryPromise(async (signal) => {
    //   const response = await fetch("http://localhost:8000/detect", { method: "POST", body, signal });
    //   if (!response.ok) throw new Error(`Vision request failed: ${response.status} ${response.statusText}`);
    //   return response.json();
    // });

    yield* Effect.sleep("100 millis");
    const result = "Mock detections; no measured heights.";
    yield* Console.log("[vision:live] detectWithHeights — finished");
    return result;
  }),
});
