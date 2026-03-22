import type { IpcMain } from "electron";

export function registerRuntimeIpcHandlers({
  ipcMain,
  getGpuDevicesCached,
  getSystemRuntimeInfoCached,
}: {
  ipcMain: IpcMain;
  getGpuDevicesCached: () => Promise<{ data: any; fromCache: boolean }>;
  getSystemRuntimeInfoCached: () => Promise<any>;
}) {
  ipcMain.handle("get-gpu-devices", async () => {
    const gpu = await getGpuDevicesCached();
    return gpu.data;
  });

  ipcMain.handle("get-system-runtime-info", async () => {
    return getSystemRuntimeInfoCached();
  });
}
