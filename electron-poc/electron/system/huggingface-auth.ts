import { BrowserWindow, Menu, app, shell } from "electron";
import path from "path";

export function createHuggingFaceAuthWindowController({
  getMainWindow,
}: {
  getMainWindow: () => BrowserWindow | null;
}) {
  let hfAuthWindow: BrowserWindow | null = null;

  function openHuggingFaceAuthWindow() {
    if (hfAuthWindow && !hfAuthWindow.isDestroyed()) {
      hfAuthWindow.focus();
      return;
    }

    hfAuthWindow = new BrowserWindow({
      width: 520,
      height: 360,
      resizable: false,
      minimizable: false,
      maximizable: false,
      modal: !!getMainWindow(),
      parent: getMainWindow() || undefined,
      title: "Authorize Hugging Face",
      webPreferences: {
        preload: path.join(__dirname, "preload.cjs"),
        nodeIntegration: false,
        contextIsolation: true,
      },
    });

    hfAuthWindow.on("closed", () => {
      hfAuthWindow = null;
    });

    const html = `<!doctype html>
  <html>
    <head>
      <meta charset="utf-8" />
      <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';" />
      <title>Authorize Hugging Face</title>
      <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 18px; color: #111; }
        h1 { font-size: 18px; margin: 0 0 8px; }
        p { margin: 8px 0; line-height: 1.35; color: #333; }
        .box { border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: #fafafa; }
        label { display: block; font-weight: 600; margin: 12px 0 6px; }
        input { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #ccc; font-size: 13px; }
        .row { display: flex; gap: 10px; margin-top: 12px; }
        button { padding: 10px 12px; border-radius: 8px; border: 1px solid #bbb; background: white; cursor: pointer; font-weight: 600; }
        button.primary { background: #1677ff; border-color: #1677ff; color: white; }
        button.danger { background: #fff; border-color: #d33; color: #d33; }
        .status { margin-top: 10px; font-size: 12px; color: #444; }
        .hint { font-size: 12px; color: #555; }
        a { color: #1677ff; text-decoration: none; }
      </style>
    </head>
    <body>
      <h1>Authorize Hugging Face (optional)</h1>
      <p class="hint">Only needed for gated/private model downloads. Public models work without this.</p>
      <div class="box">
        <div id="current" class="status">Checking status…</div>
        <label for="token">Token</label>
        <input id="token" type="password" placeholder="Paste your Hugging Face access token" autocomplete="off" />
        <div class="row">
          <button class="primary" id="save">Save token</button>
          <button class="danger" id="clear">Clear token</button>
          <button id="close">Close</button>
        </div>
        <p class="hint">Create a token in <a href="#" id="openTokens">huggingface.co/settings/tokens</a>.</p>
        <div id="msg" class="status"></div>
      </div>
      <script>
        const current = document.getElementById('current');
        const msg = document.getElementById('msg');
        const tokenEl = document.getElementById('token');

        async function refresh() {
          try {
            const st = await window.electronAPI.getHuggingFaceAuthStatus();
            current.textContent = st && st.configured ? 'Status: Token configured' : 'Status: Not configured';
          } catch (e) {
            current.textContent = 'Status: Unknown';
          }
        }

        document.getElementById('openTokens').addEventListener('click', async (e) => {
          e.preventDefault();
          await window.electronAPI.openExternalUrl('https://huggingface.co/settings/tokens');
        });

        document.getElementById('save').addEventListener('click', async () => {
          msg.textContent = '';
          const t = tokenEl.value || '';
          const res = await window.electronAPI.setHuggingFaceToken(t);
          if (res && res.success) {
            tokenEl.value = '';
            msg.textContent = 'Saved. Backend will restart to apply.';
            await refresh();
          } else {
            msg.textContent = (res && res.error) ? res.error : 'Failed to save token.';
          }
        });

        document.getElementById('clear').addEventListener('click', async () => {
          msg.textContent = '';
          const res = await window.electronAPI.clearHuggingFaceToken();
          if (res && res.success) {
            msg.textContent = 'Cleared. Backend will restart to apply.';
            await refresh();
          } else {
            msg.textContent = (res && res.error) ? res.error : 'Failed to clear token.';
          }
        });

        document.getElementById('close').addEventListener('click', () => window.close());
        refresh();
      </script>
    </body>
  </html>`;

    hfAuthWindow.loadURL(
      `data:text/html;charset=utf-8,${encodeURIComponent(html)}`,
    );
  }

  return {
    openHuggingFaceAuthWindow,
  };
}

export function installApplicationMenu({
  log,
  openHuggingFaceAuthWindow,
  setStoredHuggingFaceToken,
  requestBridgeRestart,
}: {
  log: (message: string, ...args: any[]) => void;
  openHuggingFaceAuthWindow: () => void;
  setStoredHuggingFaceToken: (token: string | null) => { success: boolean; error?: string };
  requestBridgeRestart: (reason: string) => void;
}) {
  try {
    const template: Electron.MenuItemConstructorOptions[] = [
      ...(process.platform === "darwin"
        ? ([
            {
              label: app.name,
              submenu: [
                { role: "about" },
                { type: "separator" },
                { role: "quit" },
              ],
            },
          ] as Electron.MenuItemConstructorOptions[])
        : []),
      {
        label: "File",
        submenu: [
          process.platform === "darwin" ? { role: "close" } : { role: "quit" },
        ],
      },
      {
        label: "Hugging Face",
        submenu: [
          {
            label: "Authorize Hugging Face…",
            click: () => openHuggingFaceAuthWindow(),
          },
          {
            label: "Clear Hugging Face Token",
            click: () => {
              const res = setStoredHuggingFaceToken(null);
              if (res.success) requestBridgeRestart("cleared huggingface token");
            },
          },
          { type: "separator" },
          {
            label: "Open Token Settings…",
            click: async () => {
              await shell.openExternal("https://huggingface.co/settings/tokens");
            },
          },
        ],
      },
    ];
    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
  } catch (e) {
    log("Failed to set application menu:", e);
  }
}
