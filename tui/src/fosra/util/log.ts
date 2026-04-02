import { Logger } from "tslog";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

const LOG_FILE = path.join(os.homedir(), ".fosra-tui.log");

function ensureLogDir() {
  const dir = path.dirname(LOG_FILE);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

const logger = new Logger({
  name: "forsra-tui",
  minLevel: 2,
  type: "pretty",
  prettyLogTemplate: "{{dateIsoStr}}\t{{logLevelName}}\t{{name}}\t{{msg}}\t{{args}}",
  prettyLogTimeZone: "local",
  stylePrettyLogs: true,
  prettyLogStyles: {
    logLevelName: {
      SILLY: ["bold", "white"],
      TRACE: ["bold", "whiteBright"],
      DEBUG: ["bold", "cyan"],
      INFO: ["bold", "green"],
      WARN: ["bold", "yellow"],
      ERROR: ["bold", "red"],
      FATAL: ["bold", "redBright"],
    },
    dateIsoStr: "dim",
    name: "cyan",
  },
});

logger.attachTransport((logObj) => {
  ensureLogDir();
  const line = JSON.stringify(logObj, null, 2) + "\n";
  fs.appendFileSync(LOG_FILE, line);
});

export const Log = {
  Default: logger,
};
