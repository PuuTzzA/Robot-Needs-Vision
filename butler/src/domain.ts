import { Schema } from "effect";

export const Destination = Schema.Literals(["E1", "E2", "M1", "M2", "D1", "D2"]);
export type Destination = typeof Destination.Type;
export type Position = Destination | "home";

// Map-frame position in metres, heading in degrees.
export const Coordinate = Schema.Struct({
  x: Schema.Finite,
  y: Schema.Finite,
  yaw_degrees: Schema.Finite.check(
    Schema.isBetween({ minimum: 0, maximum: 360 }),
  ),
});
export type Coordinate = typeof Coordinate.Type;

// Replace these placeholders with measured poses before enabling live transport.
export const coordinates: Readonly<Record<Position, Coordinate>> = {
  home: { x: 0, y: 0, yaw_degrees: 0 },
  E1: { x: 0, y: 0, yaw_degrees: 0 },
  E2: { x: 0, y: 0, yaw_degrees: 0 },
  M1: { x: 0, y: 0, yaw_degrees: 0 },
  M2: { x: 0, y: 0, yaw_degrees: 0 },
  D1: { x: 0, y: 0, yaw_degrees: 0 },
  D2: { x: 0, y: 0, yaw_degrees: 0 },
};
