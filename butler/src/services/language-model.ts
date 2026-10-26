import { OpenAiClient, OpenAiLanguageModel } from "@effect/ai-openai";
import { Config, Console, Effect, Layer, Stream } from "effect";
import { LanguageModel } from "effect/ai";
import { FetchHttpClient } from "effect/http";

export const MockLanguageModel = Layer.effect(
  LanguageModel.LanguageModel,
  LanguageModel.make({
    generateText: Effect.fnUntraced(function*(options) {
      const planning = options.responseFormat.type === "json";
      const operation = planning ? "plan" : "generateText";
      yield* Console.log(`[llm:mock] ${operation} — started`);
      yield* Effect.sleep("400 millis");
      yield* Console.log(`[llm:mock] ${operation} — finished`);
      return [{
        type: "text" as const,
        // Two fixed points demonstrate analysis overlapping the next drive.
        text: planning ? '{"destinations":["E1","M1"]}' : "Mock answer.",
      }];
    }),
    streamText: () => Stream.empty,
  }),
);

// Real provider; requires OPENAI_API_KEY and a live camera for visual tasks.
export const LiveLanguageModel = OpenAiLanguageModel.model("gpt-6-luna").pipe(
  Layer.provide(OpenAiClient.layerConfig({
    apiKey: Config.Redacted("OPENAI_API_KEY"),
  }).pipe(Layer.provide(FetchHttpClient.layer))),
);
