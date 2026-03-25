export type ModelCatalogTier =
  | "verified"
  | "advanced_manual"
  | "advanced"
  | "manual"
  | "blocked"
  | "online"
  | string;

export type CatalogSelectionType = "model" | "recipe" | "workflow";

export type ModelSourceKind =
  | "guide"
  | "vendor"
  | "community"
  | "mirror"
  | "local"
  | "browser"
  | string;

export type ModelInstallPolicy =
  | "direct"
  | "manual"
  | "hybrid"
  | "custom_runtime"
  | "unavailable"
  | string;

export interface ModelVerificationMetadata {
  lastVerified?: string;
  verifiedAt?: string;
  verifiedBy?: string;
  source?: string;
  evidence?: string;
  notes?: string[];
}

export interface ModelSelectionEnvelope {
  selectionType?: CatalogSelectionType;
  selectionId?: string;
  catalogTier?: ModelCatalogTier;
  sourceKind?: ModelSourceKind;
  installPolicy?: ModelInstallPolicy;
  verification?: ModelVerificationMetadata;
}

export interface ModelCatalogMetadata {
  tier?: ModelCatalogTier;
  sourceKind?: ModelSourceKind;
  installPolicy?: ModelInstallPolicy;
  verification?: ModelVerificationMetadata;
  selectionType?: CatalogSelectionType;
  selectionId?: string;
}

export interface CatalogSelectionRecord {
  selection_type?: CatalogSelectionType;
  selection_id?: string;
  selectionType?: CatalogSelectionType;
  selectionId?: string;
  name?: string;
  catalog_tier?: ModelCatalogTier;
  catalogTier?: ModelCatalogTier;
  source_kind?: ModelSourceKind;
  sourceKind?: ModelSourceKind;
  install_policy?: ModelInstallPolicy;
  installPolicy?: ModelInstallPolicy;
  runtime_adapter?: string;
  runtimeAdapter?: string;
  required_model_ids?: string[];
  requiredModelIds?: string[];
  selection_envelope?: ModelSelectionEnvelope;
  selectionEnvelope?: ModelSelectionEnvelope;
  verification?: ModelVerificationMetadata;
  [key: string]: unknown;
}

export interface CatalogRuntimeManifest {
  version?: string;
  schema_version?: string;
  generated_at?: string;
  source?: Record<string, unknown>;
  summary?: Record<string, unknown>;
  catalog_status?: CatalogStatus;
  models?: Record<string, unknown>[];
  recipes?: Record<string, unknown>[];
  workflows?: Record<string, unknown>[];
  external_records?: Record<string, unknown>[];
  selection_index?: CatalogSelectionRecord[];
}

export interface CatalogStatus {
  active_revision?: string;
  source_url?: string;
  fetched_at?: string | null;
  fallback_kind?: string;
  stale?: boolean;
  signature_valid?: boolean;
  active_path?: string;
}

export interface SelectionInstallPlan {
  selectionType: CatalogSelectionType;
  selectionId: string;
  name?: string;
  catalogTier?: ModelCatalogTier;
  sourceKind?: ModelSourceKind;
  installPolicy?: ModelInstallPolicy;
  runtimeAdapter?: string;
  requiredModelIds?: string[];
  selectionEnvelope?: ModelSelectionEnvelope;
  verification?: ModelVerificationMetadata;
  installation?: Record<string, unknown>;
  download?: Record<string, unknown>;
  model?: Record<string, unknown>;
  recipe?: Record<string, unknown>;
  workflow?: Record<string, unknown>;
  [key: string]: unknown;
}
