// src/hooks/debug.ts
import { useEffect, useRef } from "react";

export function useWhyDidYouUpdate(name: string, props: Record<string, any>) {
  const previousProps = useRef<Record<string, any>>();

  useEffect(() => {
    if (previousProps.current) {
      const allKeys = Object.keys({ ...previousProps.current, ...props });
      const changedProps: Record<string, { from: any; to: any }> = {};

      allKeys.forEach((key) => {
        if (previousProps.current![key] !== props[key]) {
          changedProps[key] = {
            to: props[key],
            from: previousProps.current![key],
          };
        }
      });

      if (Object.keys(changedProps).length > 0) {
        console.group(
          `%c[${name}] Props changed`,
          "color: orange; font-weight: bold",
        );

        console.table(changedProps);
        console.groupEnd();
      }
    }

    previousProps.current = props;
  });
}

export function useRenderCount(componentName: string) {
  const renders = useRef(0);

  useEffect(() => {
    renders.current += 1;
    console.log(`[${componentName}] Render #${renders.current}`);
  });
}

// Enhanced version with type checking
export function useWhyDidYouUpdateAdvanced(
  name: string,
  props: Record<string, any>,
) {
  const previousProps = useRef<Record<string, any>>();
  const renderCount = useRef(0);

  useEffect(() => {
    renderCount.current += 1;

    if (previousProps.current) {
      const allKeys = Object.keys({ ...previousProps.current, ...props });
      const changedProps: Record<
        string,
        {
          from: any;
          to: any;
          type: string;
          reason: string;
        }
      > = {};

      allKeys.forEach((key) => {
        const prevValue = previousProps.current![key];
        const currValue = props[key];

        if (prevValue !== currValue) {
          let reason = "different reference";

          // Detect why it changed
          if (
            typeof prevValue === "function" &&
            typeof currValue === "function"
          ) {
            reason = "function recreated (missing useCallback?)";
          } else if (Array.isArray(prevValue) && Array.isArray(currValue)) {
            if (JSON.stringify(prevValue) === JSON.stringify(currValue)) {
              reason = "array recreated with same values (missing useMemo?)";
            } else {
              reason = "array contents changed";
            }
          } else if (
            typeof prevValue === "object" &&
            typeof currValue === "object" &&
            prevValue !== null &&
            currValue !== null
          ) {
            if (JSON.stringify(prevValue) === JSON.stringify(currValue)) {
              reason = "object recreated with same values (missing useMemo?)";
            } else {
              reason = "object contents changed";
            }
          } else if (prevValue !== currValue) {
            reason = "value changed";
          }

          changedProps[key] = {
            from: prevValue,
            to: currValue,
            type: typeof currValue,
            reason,
          };
        }
      });

      if (Object.keys(changedProps).length > 0) {
        console.group(
          `%c[why-did-you-update] ${name} (render #${renderCount.current})`,
          "color: orange; font-weight: bold",
        );
        console.log("Changed props:", changedProps);
        console.table(changedProps);
        console.groupEnd();
      }
    }

    previousProps.current = props;
  });
}

export function printObjTypes(
  obj,
  label = "",
  depth = 0,
  maxDepth = 100,
  seen = new WeakSet(),
) {
  const indent = "  ".repeat(depth);

  if (obj === null) {
    console.log(`${indent}${label}: %cnull`, "color: #999");
    return;
  }
  if (obj === undefined) {
    console.log(`${indent}${label}: %cundefined`, "color: #999");
    return;
  }

  const type = typeof obj;
  if (type !== "object") {
    const color =
      type === "string" ? "#ce9178" : type === "number" ? "#b5cea8" : "#569cd6";
    console.log(`${indent}${label}: %c${type}`, `color: ${color}`);
    return;
  }

  if (seen.has(obj)) {
    console.log(`${indent}${label}: %c[Circular]`, "color: #f48771");
    return;
  }
  seen.add(obj);

  if (Array.isArray(obj)) {
    console.log(
      `${indent}${label}: %cArray(${obj.length})`,
      "color: #4ec9b0; font-weight: bold",
    );
    if (depth < maxDepth) {
      obj.forEach((item, i) => {
        printObjTypes(item, `[${i}]`, depth + 1, maxDepth, seen);
      });
    }
    return;
  }

  const ctorName = obj.constructor?.name || "Object";
  console.log(
    `${indent}${label}: %c${ctorName}`,
    "color: #4ec9b0; font-weight: bold",
  );

  if (depth < maxDepth) {
    Object.entries(obj).forEach(([key, val]) => {
      printObjTypes(val, key, depth + 1, maxDepth, seen);
    });
  }
}

export function printTypes(...args) {
  const seen = new WeakSet();

  const inspect = (obj, label = "", depth = 0, maxDepth = 100) => {
    const indent = "  ".repeat(depth);

    if (obj === null) {
      console.log(`${indent}${label}: %cnull`, "color: #999");
      return;
    }
    if (obj === undefined) {
      console.log(`${indent}${label}: %cundefined`, "color: #999");
      return;
    }

    const type = typeof obj;
    if (type !== "object") {
      const color =
        type === "string"
          ? "#ce9178"
          : type === "number"
            ? "#b5cea8"
            : "#569cd6";
      console.log(`${indent}${label}: %c${type}`, `color: ${color}`);
      return;
    }

    if (seen.has(obj)) {
      console.log(`${indent}${label}: %c[Circular]`, "color: #f48771");
      return;
    }
    seen.add(obj);

    if (Array.isArray(obj)) {
      console.log(
        `${indent}${label}: %cArray(${obj.length})`,
        "color: #4ec9b0; font-weight: bold",
      );
      if (depth < maxDepth) {
        obj.forEach((item, i) => {
          inspect(item, `[${i}]`, depth + 1, maxDepth);
        });
      }
      return;
    }

    const ctorName = obj.constructor?.name || "Object";
    console.log(
      `${indent}${label}: %c${ctorName}`,
      "color: #4ec9b0; font-weight: bold",
    );

    if (depth < maxDepth) {
      Object.entries(obj).forEach(([key, val]) => {
        inspect(val, key, depth + 1, maxDepth);
      });
    }
  };

  args.forEach((arg, i) => {
    console.log(
      `\n%c=== Argument ${i} ===`,
      "color: #dcdcaa; font-weight: bold; font-size: 14px",
    );
    inspect(arg, "");
  });
}
