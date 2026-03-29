// Stub for @/provider/provider
export interface ParsedModel {
  providerID: string
  modelID: string
  variant?: string
}

export const Provider = {
  parseModel: (model: string): ParsedModel => {
    const parts = model.split("/")
    if (parts.length === 2) {
      return { providerID: parts[0], modelID: parts[1] }
    }
    return { providerID: "openai", modelID: model }
  },
}