import {
  createFileRoute,
  Navigate,
  redirect,
  useNavigate,
} from "@tanstack/react-router";
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
import {
  checkUserExistUsersLoginCheckUserGet,
  CheckUserExistUsersLoginCheckUserGetData,
  validateUserLoginUsersLoginUserLoginGet,
} from "../lib/api";
import { useNavStore } from "../hooks/state-hooks";
import { queryClient } from "../main";
import { Switch } from "../components/ui/switch";

export const Route = createFileRoute("/login")({
  beforeLoad: async ({ location }) => {
    if (checkLoggedIn()) {
      throw redirect({
        to: "/convo",
        search: {
          redirect: location.href,
        },
      });
    }
  },
  component: LoginForm,
});

const checkLoggedIn = () => {
  const { activeUserId } = useNavStore.getState();

  console.log(`checkloggedin`, activeUserId,);
  if (activeUserId) {
    return true;
  } else {
    return false;
  }
};

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

  const { setActiveUserId, activeUserId } = useNavStore.getState();
  const navigate = useNavigate();

  const form = useForm({
    defaultValues: {
      Username: "",
      Password: "",
      Enabled: true,
    },
    validators: {
      onSubmitAsync: async ({ formApi, value }) => {
        const get = await validateUserLoginUsersLoginUserLoginGet({
          query: {
            Username: value.Username,
            Password: value.Password,
            Enabled: value.Enabled,
          },
        });

        const user = get.data;

        if (user) {
          setActiveUserId(user.UserId);
          return undefined;
        } else {
          setActiveUserId("");
          return "incorrect password or username";
        }
      },
    },

    onSubmit: async ({ formApi, value }) => {
      await queryClient.invalidateQueries();

      formApi.reset();

      navigate({ to: "/convo" });
    },
  });

  if (isLoading) return <p>Loading..</p>;

  return (
    <div>
      <h1>LOGIN</h1>
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
                  ? "A first name is required"
                  : value.length < 3
                    ? "First name must be at least 3 characters"
                    : undefined,
              onChangeAsyncDebounceMs: 500,
              onChangeAsync: async ({ value }) => {
                const get = await checkUserExistUsersLoginCheckUserGet({
                  query: { username: value },
                });
                const valid = get.data;
                console.log(valid);
                return valid === true ? "User Doesn't Exist" : undefined;
              },
            }}
            children={(field) => {
              // Avoid hasty abstractions. Render props are great!
              return (
                <>
                  <label htmlFor={field.name}>{field.name}</label>
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
                  : value.length < 3
                    ? "Password must be at least 8 characters long"
                    : undefined,
            }}
            children={(field) => {
              return (
                <>
                  <label htmlFor={field.name}>{field.name}</label>
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
                <label htmlFor={field.name}>{field.name}</label>
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
