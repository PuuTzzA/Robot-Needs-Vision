import { Console, Context, Effect, Layer } from "effect";
import type { Coordinate, Position } from "../domain.ts";

export class Transport extends Context.Service<Transport, {
  // Complete only on arrival.
  readonly goTo: (destination: Position, coordinate: Coordinate) => Effect.Effect<void, Error>;
}>()("butler/Transport") {}

export const MockTransport = Layer.succeed(Transport, {
  goTo: Effect.fnUntraced(function*(destination: Position) {
    yield* Console.log(`[transport:mock] goTo(${destination}) — started`);
    yield* Effect.sleep("700 millis");
    yield* Console.log(`[transport:mock] goTo(${destination}) — finished`);
  }),
});

// Requires running Nav2/localization. Does not cancel goals on interruption.
export const LiveTransport = Layer.effect(Transport, Effect.gen(function*() {
  const { default: ros } = yield* Effect.tryPromise(() => import("rclnodejs"));
  const context = yield* Effect.acquireRelease(
    Effect.try(() => new ros.Context()),
    (context) => Effect.sync(() => ros.shutdown(context)),
  );
  yield* Effect.tryPromise(() => ros.init(context));
  const node = yield* Effect.acquireRelease(
    Effect.try(() => new ros.Node("butler_transport", "", context)),
    (node) => Effect.sync(() => node.destroy()),
  );
  const client = yield* Effect.try(() => new ros.ActionClient(node, "nav2_msgs/action/NavigateToPose", "/navigate_to_pose"));
  yield* Effect.try(() => node.spin());
  return Transport.of({
    goTo: (destination, { x, y, yaw_degrees }) => Effect.tryPromise(async () => {
      if (!client.isActionServerAvailable()) throw new Error("Nav2 is unavailable");
      const halfYaw = yaw_degrees * Math.PI / 360;
      const goal = await client.sendGoal({
        pose: {
          header: { frame_id: "map", stamp: node.getClock().now().toMsg() },
          pose: {
            position: { x, y, z: 0 },
            orientation: { x: 0, y: 0, z: Math.sin(halfYaw), w: Math.cos(halfYaw) },
          },
        },
        behavior_tree: "",
      });
      if (!goal.isAccepted()) throw new Error(`Nav2 rejected ${destination}`);
      await goal.getResult();
      if (goal.status !== 4) throw new Error(`Nav2 failed at ${destination} (${goal.status})`);
    }),
  });
}));
