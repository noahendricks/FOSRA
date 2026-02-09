import { defineConfig } from "@hey-api/openapi-ts";

export default defineConfig({
  input: "http://localhost:8000/openapi.json",
  output: "./src/lib/api",
  plugins: [
    "@hey-api/client-axios", // Modern Axios wrapper
    "zod", // Generates your Zod schemas
    "@hey-api/sdk", // Generates the "Services" you'll call
    {
      name: "@tanstack/react-query",
      queryOptions: true, // Generates the magic for useQuery
    },
  ],
});
