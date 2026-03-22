import type { IpcMain } from "electron";

type SendBackendCommand = (
  command: string,
  payload?: Record<string, any>,
  timeoutMs?: number,
) => Promise<any>;

export function registerQueueIpcHandlers({
  ipcMain,
  sendBackendCommand,
}: {
  ipcMain: IpcMain;
  sendBackendCommand: SendBackendCommand;
}) {
  ipcMain.handle("pause-queue", async () => sendBackendCommand("pause_queue"));
  ipcMain.handle("resume-queue", async () => sendBackendCommand("resume_queue"));
  ipcMain.handle("reorder-queue", async (_event, jobIds: string[]) =>
    sendBackendCommand("reorder_queue", { job_ids: jobIds }),
  );
}
