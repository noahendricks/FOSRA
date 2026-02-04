import { decodeTime } from "ulid";
import * as z from "zod";

// for error handling later
const ULID_FULL_LENGTH = 26;
const ULID_TIMESTAMP_LENGTH = 10;
const CROCKFORD_BASE32_CHARS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
const HEX_CHARS = "0123456789ABCDEF";

const defaultDateTimeFormat = new Intl.DateTimeFormat([], {
  year: "numeric",
  month: "numeric",
  day: "numeric",
  hour: "numeric",
  minute: "numeric",
  second: "numeric",
  fractionalSecondDigits: 3,
  timeZoneName: "short",
  hour12: false,
});

const list_of_ulid = new Array(
  "01KE0KFK42CB84K27C0VVCDVT2",
  "01KE0KGSKF4QTQJZPKK365BWJ8",
  "01KE0KGSKRP8AR6RBDPMR91WVF",
  "01KE0KGSKX7GZMK1HV8XBNXD15",
  "01KE0KGSM2PEVHSCPTT85FXYC5",
  "01KE0KGSM8JWHPZJZHF0JAPV0P",
  "01KE0M6F50SWYW3YFB2Y1C55N8",
  "01KE0M6F5BQ4GCSDA1H0XHGAJ0",
  "01KE0M6F5FPY2JTM5FB8W18CE7",
  "01KE0M6F5N5AGBH0WGF0QK69VB",
  "01KE0M6F5TPFZ2GB9G4QW0KWTV",
  "01KE0MB54BFJH23NQEMYZHWQVN",
  "01KE0MB54J6GV0SH26XCSP9SVT",
  "01KE0MB54RS0VD51GFW1767BQ4",
  "01KE0MB54Y5F0027H67VK709D4",
  "01KE0MB552MJ3WJZCB2VNGA7WV",
  "01KE0MB65JGY3M82QQ40K95BPH",
  "01KE0MB65VXYW469110EG53FFQ",
  "01KE0MB65Z318D5GHCBDQNNT5X",
  "01KE0MB663CM926E86QPY2E1VY",
  "01KE0MB669C0QSY9A724NBJYHE",
  "01KE0MB8A91QN4M1DC5XVS9R7Q",
);

export const ulidDateInfo = z.object({
  local_date: z.string(),
  local_date_numeric: z.string(),
  date_iso: z.iso.datetime(),
  og_id: z.string(),
});

export function ulidToDateInfo(inUlid: string) {
  // Add error handling
  const epochMs = decodeTime(inUlid);

  const dt = new Date(epochMs);
  // const epochInMs = epochMs.toString();
  const dateLocalDefault = dt.toString();
  const dateLocalNumeric = defaultDateTimeFormat.format(dt);
  const dateUtcISO = dt.toISOString();


  const data = {
    local_date: dateLocalDefault,
    local_date_numeric: dateLocalNumeric,
    date_iso: dateUtcISO,
    og_id: inUlid
  }

  return ulidDateInfo.parse(data)

}

export function ulidListToSortedInfoList(ulidList: string[]) {
  const sorted_list = ulidList.map((id) => ulidToDateInfo(id));
  return sorted_list.sort();
}

// console.log(ulidListToSortedInfoList(list_of_ulid));
