import { Console, Context, Effect, Layer } from "effect";
import { coordinates, type Coordinate, type Destination } from "../domain.ts";

export type Route = ReadonlyArray<readonly [Destination, Coordinate]>;

const locate = (points: ReadonlySet<Destination>): Route =>
  [...points].map((destination) => [destination, coordinates[destination]] as const);

export class Navigation extends Context.Service<Navigation, {
  readonly order: (points: ReadonlySet<Destination>) => Effect.Effect<Route, Error>;
}>()("butler/Navigation") {}

export const MockNavigation = Layer.succeed(Navigation, {
  order: Effect.fnUntraced(function*(points: ReadonlySet<Destination>) {
    yield* Console.log("[navigation:mock] order — started");
    yield* Effect.sleep("50 millis");
    yield* Console.log("[navigation:mock] order — finished");
    return locate(points);
  }),
});

export const LiveNavigation = Layer.succeed(Navigation, {
  order: Effect.fnUntraced(function*(points: ReadonlySet<Destination>) {
    yield* Console.log(`[navigation] order(${[...points].join(", ")})`);
    return locate(points);
  }),
});

// TODO: Order every requested point using the current level's map and travel costs.
export const LiveAdvancedNavigation = Layer.succeed(Navigation, {
  order: (points) => Console.log(`[navigation] order(${[...points].join(", ")})`).pipe(
    Effect.andThen(Effect.fail(new Error("Advanced navigation is not implemented"))),
  ),
});
