import { describe, expect, it } from "vitest"
import { toMediaUrl } from "@/lib/media/toMediaUrl"

describe("toMediaUrl", () => {
  it("normalizes Windows-style paths into the media protocol", () => {
    expect(toMediaUrl("C:\\Audio Files\\vocals stem.wav")).toBe(
      "media://C:/Audio%20Files/vocals%20stem.wav",
    )
  })

  it("preserves already normalized unix-style paths", () => {
    expect(toMediaUrl("/tmp/stems/instrumental.wav")).toBe(
      "media:///tmp/stems/instrumental.wav",
    )
  })
})
