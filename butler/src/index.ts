import { BunRuntime } from "@effect/platform-bun";
import { Effect, Layer } from "effect";
import { runLevel } from "./program.ts";
import { MockCamera } from "./services/camera.ts";
import { LiveLanguageModel, MockLanguageModel } from "./services/language-model.ts";
import { LiveNavigation } from "./services/navigation.ts";
import { MockTransport } from "./services/transport.ts";
import { MockVision } from "./services/vision.ts";

// Swap individual imports/layers for their Mock or Live counterparts.
const services = Layer.mergeAll(
  LiveNavigation,
  MockTransport,
  MockCamera,
  MockVision,
  MockLanguageModel,
);

BunRuntime.runMain(runLevel.pipe(Effect.provide(services)));
