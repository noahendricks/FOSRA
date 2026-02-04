import { createFileRoute, redirect } from "@tanstack/react-router";
import * as React from "react";

import { formDevtoolsPlugin } from "@tanstack/react-form-devtools";
import { useForm } from "@tanstack/react-form";
import {
  QueryClient,
  QueryClientProvider,
  useMutation,
  useQuery,
} from "@tanstack/react-query";

import type { AnyFieldApi } from "@tanstack/react-form";
import { newUserProfileUsersCreateUserPostMutation } from "../lib/api/@tanstack/react-query.gen";
import {
  checkUserExistUsersLoginCheckUserGet,
  CheckUserExistUsersLoginCheckUserGetData,
} from "../lib/api";
import { useNavStore } from "../hooks/state-hooks";
import { queryClient } from "../main";
import { Switch } from "../components/ui/switch";

export const Route = createFileRoute("/create_user")({
  component: LoginForm,
});

function FieldInfo({ field }: { field: AnyFieldApi }) {
  return (
    <>
      {field.state.meta.isTouched && !field.state.meta.isValid ? (
        <em>{field.state.meta.errors.join(",")}</em>
      ) : null}
      {field.state.meta.isValidating ? "Validating..." : null}
    </>
  );
}

export default function LoginForm() {
  const [isLoading, setLoading] = React.useState();

  const createUser = useMutation(newUserProfileUsersCreateUserPostMutation());

  const { setActiveUserId, activeUserId } = useNavStore.getState();

  const form = useForm({
    defaultValues: {
      Username: "",
      Password: "",
      Enabled: true,
    },
    onSubmit: async ({ formApi, value }) => {
      const out = await createUser.mutateAsync({
        body: {
          Username: value.Username,
          Password: value.Password,
          Enabled: value.Enabled,
        },
      });

      setActiveUserId(out.UserId);

      await queryClient.invalidateQueries();
      // Reset the form to start-over with a clean state
      formApi.reset();
      throw redirect({ to: "/convo" });
    },
  });

  if (isLoading) return <p>Loading..</p>;

  return (
    <div>
      <h1>Create Profile</h1>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          e.stopPropagation();
          form.handleSubmit();
        }}
      >
        <div>
          {/* A type-safe field component*/}
          <form.Field
            name="Username"
            validators={{
              onChange: ({ value }) =>
                !value
                  ? "A Username is required"
                  : value.length < 3
                    ? "Username must be at least 3 characters"
                    : undefined,
              onChangeAsyncDebounceMs: 500,
              onChangeAsync: async ({ value }) => {
                const get = await checkUserExistUsersLoginCheckUserGet({
                  query: { username: value },
                });
                const valid = get.data;
                return valid === false ? "Name is taken" : undefined;
              },
            }}
            children={(field) => {
              // Avoid hasty abstractions. Render props are great!
              return (
                <>
                  <label htmlFor={field.name}>Username</label>
                  <input
                    id={field.name}
                    name={field.name}
                    value={field.state.value}
                    onBlur={field.handleBlur}
                    onChange={(e) => field.handleChange(e.target.value)}
                    className="border border-r-white rounded-md mx-3"
                  />
                  <FieldInfo field={field} />
                </>
              );
            }}
          />
        </div>
        <div>
          {/* A type-safe field component*/}
          <form.Field
            name="Password"
            validators={{
              onChange: ({ value }) =>
                !value
                  ? "A Password is Required"
                  : value.length < 8
                    ? "Pasword must be at least 8 characters"
                    : undefined,
              onChangeAsyncDebounceMs: 500,
              onChangeAsync: async ({ value }) => { },
            }}
            children={(field) => {
              // Avoid hasty abstractions. Render props are great!
              return (
                <>
                  <label htmlFor={field.name}>Password</label>
                  <input
                    id={field.name}
                    name={field.name}
                    value={field.state.value}
                    onBlur={field.handleBlur}
                    onChange={(e) => field.handleChange(e.target.value)}
                    className="border border-r-white rounded-md mx-3"
                  />
                  <FieldInfo field={field} />
                </>
              );
            }}
          />
        </div>

        <div>
          <form.Field
            name="Enabled"
            children={(field) => (
              <>
                <label htmlFor={field.name}>Enabled</label>
                <Switch
                  checked={field.state.value}
                  onCheckedChange={(checked) => field.handleChange(checked)}
                  className=" border mx-3"
                />
                <FieldInfo field={field} />
              </>
            )}
          />
        </div>
        <form.Subscribe
          selector={(state) => [state.canSubmit, state.isSubmitting]}
          children={([canSubmit, isSubmitting]) => (
            <>
              <button type="submit" disabled={!canSubmit}>
                {isSubmitting ? "..." : "Submit"}
              </button>
              <button
                className="bg-blend-color-dodge border border-orange-700"
                type="reset"
                onClick={() => form.reset()}
              >
                Reset
              </button>
            </>
          )}
        />
      </form>
    </div>
  );
}
