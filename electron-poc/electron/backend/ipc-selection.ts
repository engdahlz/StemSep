import type { IpcMain } from "electron";

type SendBackendCommand = (
  command: string,
  payload?: Record<string, any>,
  timeoutMs?: number,
) => Promise<any>;

type RegisterSelectionIpcArgs = {
  ipcMain: IpcMain;
  sendBackendCommand: SendBackendCommand;
};

export function registerSelectionIpcHandlers({
  ipcMain,
  sendBackendCommand,
}: RegisterSelectionIpcArgs) {
  ipcMain.handle("get-catalog", async () => {
    return sendBackendCommand("get_catalog", {}, 30000);
  });

  ipcMain.handle("get-selection-installation", async (_event, payload) => {
    return sendBackendCommand(
      "get_selection_installation",
      {
        selection_type: payload.selectionType,
        selection_id: payload.selectionId,
      },
      30000,
    );
  });

  ipcMain.handle("resolve-install-plan", async (_event, payload) => {
    return sendBackendCommand(
      "resolve_install_plan",
      {
        selection_type: payload.selectionType,
        selection_id: payload.selectionId,
      },
      30000,
    );
  });

  ipcMain.handle("install-selection", async (_event, payload) => {
    return sendBackendCommand(
      "install_selection",
      {
        selection_type: payload.selectionType,
        selection_id: payload.selectionId,
        ...(payload.options || {}),
      },
      60 * 60 * 1000,
    );
  });

  ipcMain.handle("import-selection-artifacts", async (_event, payload) => {
    return sendBackendCommand(
      "import_selection_artifacts",
      {
        selection_type: payload.selectionType,
        selection_id: payload.selectionId,
        files: payload.files,
        allow_copy: payload.allowCopy !== false,
      },
      60 * 1000,
    );
  });

  ipcMain.handle("verify-selection-artifacts", async (_event, payload) => {
    return sendBackendCommand(
      "verify_selection_artifacts",
      {
        selection_type: payload.selectionType,
        selection_id: payload.selectionId,
      },
      30000,
    );
  });

  ipcMain.handle("resolve-execution-plan", async (_event, payload) => {
    return sendBackendCommand(
      "resolve_execution_plan",
      {
        selection_type: payload.selectionType,
        selection_id: payload.selectionId,
        config: payload.config || {},
      },
      30000,
    );
  });

  ipcMain.handle("run-selection-job", async (_event, payload) => {
    return sendBackendCommand("run_selection_job", payload || {}, 30000);
  });

  ipcMain.handle("cancel-selection-job-direct", async (_event, payload) => {
    return sendBackendCommand(
      "cancel_selection_job",
      { job_id: payload.jobId },
      15000,
    );
  });

  ipcMain.handle("get-selection-job", async (_event, payload) => {
    return sendBackendCommand(
      "get_selection_job",
      { job_id: payload.jobId },
      15000,
    );
  });

  ipcMain.handle("list-selection-jobs", async () => {
    return sendBackendCommand("list_selection_jobs", {}, 15000);
  });

  ipcMain.handle("export-selection-job", async (_event, payload) => {
    return sendBackendCommand(
      "export_selection_job",
      {
        job_id: payload.jobId,
        export_path: payload.exportPath,
      },
      120000,
    );
  });

  ipcMain.handle("discard-selection-job", async (_event, payload) => {
    return sendBackendCommand(
      "discard_selection_job",
      { job_id: payload.jobId },
      30000,
    );
  });
}
