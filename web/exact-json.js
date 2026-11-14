"use strict";

// Keep integer tokens from the service exact, including values too large for Number.
function readRecord(text) {
  return JSON.parse(text, (_key, value, context) => {
    if (typeof value !== "number" || Number.isSafeInteger(value)) return value;
    if (context && typeof JSON.rawJSON === "function") {
      return /^-?\d+$/.test(context.source) ? BigInt(context.source) : value;
    }
    if (!Number.isFinite(value) || Number.isInteger(value)) {
      const error = new Error("This result needs a browser that supports exact JSON integers.");
      error.name = "ExactIntegerError";
      throw error;
    }
    return value;
  });
}
