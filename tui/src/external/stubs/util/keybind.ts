// Stub for @/util/keybind
export interface KeybindInfo {
  key: string
  ctrl?: boolean
  alt?: boolean
  shift?: boolean
  meta?: boolean
  leader?: boolean
}

export const Keybind = {
  parse: (keybind: string): KeybindInfo[] => {
    const result: KeybindInfo[] = []
    const parts = keybind.split(" ")
    let leader = false
    
    for (const part of parts) {
      if (part === "<leader>") {
        leader = true
        continue
      }
      
      const info: KeybindInfo = { key: part, leader }
      
      if (part.startsWith("<C-") && part.endsWith(">")) {
        info.key = part.slice(3, -1)
        info.ctrl = true
      } else if (part.startsWith("<M-") && part.endsWith(">")) {
        info.key = part.slice(3, -1)
        info.alt = true
      } else if (part.startsWith("<S-") && part.endsWith(">")) {
        info.key = part.slice(3, -1)
        info.shift = true
      }
      
      result.push(info)
    }
    
    return result
  },
  match: (a: KeybindInfo, b: KeybindInfo): boolean => {
    return a.key === b.key && 
           !!a.ctrl === !!b.ctrl &&
           !!a.alt === !!b.alt &&
           !!a.shift === !!b.shift &&
           !!a.leader === !!b.leader
  },
  fromParsedKey: (evt: { name: string; ctrl?: boolean; alt?: boolean; shift?: boolean; meta?: boolean }, leader: boolean): KeybindInfo => {
    return {
      key: evt.name,
      ctrl: evt.ctrl,
      alt: evt.alt,
      shift: evt.shift,
      meta: evt.meta,
      leader,
    }
  },
}