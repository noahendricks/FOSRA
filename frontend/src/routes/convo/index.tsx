import { createFileRoute, redirect } from "@tanstack/react-router";
import { Shimmer } from "../../components/ai-elements/shimmer";
import { cn } from "../../lib/utils";
import { useNavStore } from "../../hooks/state-hooks";

const WelcomeShimmer = () => {
  return (
    <div className={cn("flex justify-center items-center")}>
      <Shimmer className={"text-5xl "}>welcome</Shimmer>
    </div>
  );
};

export const Route = createFileRoute("/convo/")({
  beforeLoad: async ({ location }) => {
    const { activeUserId } = useNavStore.getState();

    if (!activeUserId) {
      throw redirect({
        to: "/login",
        search: {
          redirect: location.href,
        },
      });
    }
  },

  component: WelcomeShimmer,
});
