import { createFileRoute, Outlet, Link } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  component: IndexComponent,
});

function IndexComponent() {
  return <Outlet />;
}
