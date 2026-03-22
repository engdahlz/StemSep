import { describe, expect, it } from "vitest";
import {
  formatCardMetricValue,
  formatCatalogStatus,
  getCardMetricSlots,
} from "../lib/models/cardMetrics";
import { Model } from "../types/store";

const baseModel: Model = {
  id: "model-1",
  name: "Test Model",
  architecture: "Mel-Roformer",
  version: "",
  category: "primary",
  description: "",
  sdr: 0,
  fullness: 0,
  bleedless: 0,
  vram_required: 6,
  speed: "unknown",
  stems: [],
  file_size: 0,
  installed: false,
  downloading: false,
  downloadProgress: 0,
  recommended: false,
};

describe("cardMetrics", () => {
  it("uses card metric slots when present", () => {
    const slots = getCardMetricSlots({
      ...baseModel,
      card_metrics: {
        labels: ["VOC SDR", "INST SDR", "AVG SDR"],
        values: [11.39, 17.4, 14.4],
      },
    });
    expect(slots.map((slot) => slot.label)).toEqual([
      "VOC SDR",
      "INST SDR",
      "AVG SDR",
    ]);
    expect(formatCardMetricValue(slots[0].value)).toBe("11.39");
  });

  it("falls back to legacy metrics when card metrics are absent", () => {
    const slots = getCardMetricSlots({
      ...baseModel,
      sdr: 12.5,
      fullness: 18.25,
      bleedless: 33.75,
    });
    expect(slots.map((slot) => formatCardMetricValue(slot.value))).toEqual([
      "12.50",
      "18.25",
      "33.75",
    ]);
  });

  it("formats catalog statuses for the UI", () => {
    expect(formatCatalogStatus("verified")).toBe("Verified");
    expect(formatCatalogStatus("manual_only")).toBe("Manual");
    expect(formatCatalogStatus("candidate")).toBe("Advanced");
  });
});
