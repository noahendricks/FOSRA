import { SourceGroupResponse } from "../../lib/api";

type JSONValue =
  | string
  | number
  | boolean
  | null
  | { [key: string]: JSONValue }
  | JSONValue[];
type NestedJSONMap = Record<string, Record<string, JSONValue>>;

export function transformSourceGroups(
  groups: SourceGroupResponse[],
): NestedJSONMap {
  return groups.reduce((acc, group) => {
    const sourceId = group.Source.SourceId;

    // Initialize the inner record for this Source
    acc[sourceId] = {};

    // Map each chunk as a nested entry
    group.Chunks.forEach((item) => {
      const chunkId = item.Chunk.ChunkId;

      // We cast the chunk object to JSONValue
      // (safe because it only contains strings/numbers)
      acc[sourceId][chunkId] = {
        text: item.Chunk.Text,
        score: item.SimilarityScore,
        indices: [item.Chunk.StartIndex, item.Chunk.EndIndex],
      } as JSONValue;
    });

    // Optionally add the Source metadata as its own inner key
    if (group.Source.Metadata) {
      acc[sourceId]["_source_metadata"] = group.Source.Metadata as JSONValue;
    }

    return acc;
  }, {} as NestedJSONMap);
}
