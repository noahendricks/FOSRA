import { createContext, Show, useContext, type ParentProps } from "solid-js";

export function createSimpleContext<
  T,
  Props extends Record<string, any>,
>(input: { name: string; init: ((input: Props) => T) | (() => T) }) {
  const ctx = createContext<T>();

  return {
    provider: (props: ParentProps<Props>) => {
      const init = input.init(props);
      return (
        <Show
          // @ts-expect-error
          when={init.ready === undefined || init.ready === true}
          fallback={
            <box>
              <text>Hidden by {input.name}</text>
            </box>
          }
        >
          <ctx.Provider value={init}>{props.children}</ctx.Provider>
        </Show>
      );
    },
    use() {
      const value = useContext(ctx);
      if (!value)
        throw new Error(
          `${input.name} context must be used within a context provider`,
        );
      return value;
    },
  };
}
