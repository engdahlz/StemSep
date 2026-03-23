export type CatalogSelectionType = "model" | "recipe" | "workflow";

type SendBackendCommandWithRetry = (
  command: string,
  payload?: Record<string, any>,
  timeoutMs?: number,
  maxRetries?: number,
) => Promise<any>;

export function normalizeSelectionType(
  value: unknown,
): CatalogSelectionType | null {
  const normalized = String(value || "")
    .trim()
    .toLowerCase();
  if (
    normalized === "model" ||
    normalized === "recipe" ||
    normalized === "workflow"
  ) {
    return normalized;
  }
  return null;
}

export function normalizeSelectionEnvelope(
  value: any,
): {
  selectionType: CatalogSelectionType;
  selectionId: string;
  selectionEnvelope: Record<string, any>;
} | null {
  const selectionType = normalizeSelectionType(
    value?.selectionType ?? value?.selection_type,
  );
  const selectionId = String(
    value?.selectionId ?? value?.selection_id ?? "",
  ).trim();
  if (!selectionType || !selectionId) {
    return null;
  }
  return {
    selectionType,
    selectionId,
    selectionEnvelope: {
      ...(typeof value === "object" && value !== null ? value : {}),
      selectionType,
      selectionId,
      selection_type: selectionType,
      selection_id: selectionId,
    },
  };
}

export function createSelectionPlanResolver({
  sendBackendCommandWithRetry,
}: {
  sendBackendCommandWithRetry: SendBackendCommandWithRetry;
}) {
  async function resolveSelectionExecutionPlan(
    selectionEnvelope: any,
    config?: Record<string, any>,
  ) {
    const normalized =
      normalizeSelectionEnvelope(selectionEnvelope) ||
      normalizeSelectionEnvelope({
        selectionType: config?.selectionType ?? config?.selection_type,
        selectionId: config?.selectionId ?? config?.selection_id,
      });
    if (!normalized) {
      return null;
    }
    const executionPlan = await sendBackendCommandWithRetry(
      "resolve_execution_plan",
      {
        selection_type: normalized.selectionType,
        selection_id: normalized.selectionId,
        config: config || {},
      },
      30_000,
    );
    return executionPlan
      ? {
          ...executionPlan,
          config: config || {},
          selectionType:
            executionPlan.selectionType ??
            executionPlan.selection_type ??
            normalized.selectionType,
          selectionId:
            executionPlan.selectionId ??
            executionPlan.selection_id ??
            normalized.selectionId,
          selectionEnvelope:
            executionPlan.selectionEnvelope ??
            executionPlan.selection_envelope ??
            normalized.selectionEnvelope ??
            null,
          resolvedBundle:
            executionPlan.resolvedBundle ??
            executionPlan.resolved_bundle ??
            null,
        }
      : null;
  }

  return {
    resolveSelectionExecutionPlan,
  };
}
