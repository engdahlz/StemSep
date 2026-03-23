import { app } from "electron";
import fs from "fs";
import path from "path";

export type LogFn = (message: string, ...args: any[]) => void;

export function getLogPath() {
  return path.join(app.getPath("userData"), "app.log");
}

export function createLogger(): LogFn {
  return (message: string, ...args: any[]) => {
    const timestamp = new Date().toISOString();
    const formattedMessage = `${timestamp} - ${message} ${args.length > 0 ? JSON.stringify(args) : ""}\n`;

    console.log(message, ...args);

    try {
      fs.appendFileSync(getLogPath(), formattedMessage);
    } catch (error) {
      console.error("Failed to write to log file:", error);
    }
  };
}
