import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ModelSelector } from "../components/ModelSelector";
import { EnsembleBuilder } from "../components/EnsembleBuilder";

describe("Installed-model invariants", () => {
  it("ModelSelector only shows installed models", async () => {
    const user = userEvent.setup();
    const models: any[] = [
      { id: "m1", name: "M1", architecture: "Arch", installed: false },
      { id: "m2", name: "M2", architecture: "Arch", installed: true },
      { id: "m3", name: "M3", architecture: "Arch", installed: undefined },
    ];

    render(
      <ModelSelector
        selectedModelId={""}
        onSelectModel={() => {}}
        models={models}
      />
    );

    // open dropdown
    const btn = screen.getByRole("combobox");
    await user.click(btn);

    // Only installed model should be present.
    expect(await screen.findByText("M2")).toBeInTheDocument();
    expect(screen.queryByText("M1")).not.toBeInTheDocument();
    expect(screen.queryByText("M3")).not.toBeInTheDocument();
  });

  it("EnsembleBuilder only offers models with installed===true", () => {
    const models: any[] = [
      { id: "m1", name: "M1", installed: false },
      { id: "m2", name: "M2", installed: true },
      { id: "m3", name: "M3", installed: undefined },
    ];

    render(
      <EnsembleBuilder
        models={models}
        config={[{ model_id: "m2", weight: 1.0 }]}
        algorithm={"average" as any}
        onChange={() => {}}
      />
    );

    // The select should include M2, and not include M1/M3.
    expect(screen.getByRole("option", { name: /m2/i })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /m1/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /m3/i })).not.toBeInTheDocument();
  });
});
