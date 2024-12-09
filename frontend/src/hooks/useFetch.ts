/* eslint-disable @typescript-eslint/no-explicit-any */
import React from "react";

type FetchMethods = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

type State<T> = {
	data: T | null;
	loading: boolean;
	error: Error | null;
};

type Action<T> =
	| { type: "LOADING" }
	| { type: "SUCCESS"; payload: T }
	| { type: "ERROR"; payload: Error };

const dataFetchReducer = <T>(state: State<T>, action: Action<T>): State<T> => {
	switch (action.type) {
		case "LOADING":
			return { ...state, loading: true };
		case "SUCCESS":
			return { ...state, data: action.payload, loading: false };
		case "ERROR":
			return { ...state, error: action.payload, loading: false };
		default:
			throw new Error();
	}
};

const useFetch = <T>(
	url: string,
	method: FetchMethods = "GET",
	initialBody?: any
): {
	data: T | null;
	loading: boolean;
	error: Error | null;
	executeRequest: (body?: any, isFormData?: boolean) => Promise<void>;
} => {
	const [state, dispatch] = React.useReducer(dataFetchReducer<T>, {
		data: null,
		loading: false,
		error: null,
	});

	const executeRequest = React.useCallback(
		async (body: any = initialBody, isFormData: boolean = false) => {
			dispatch({ type: "LOADING" });

			try {
				const options: {
					method: FetchMethods;
					headers?: Record<string, string>;
					body?: string;
				} = {
					method,
				};

				if (!isFormData) {
					options.headers = {
						"Content-Type": "application/json",
					};
				}

				if (method !== "GET" && body) {
					options.body = isFormData ? body : JSON.stringify(body);
				}

				const response = await fetch(url, options);

				if (!response.ok) {
					throw new Error(`HTTP error! status: ${response.status}`);
				}

				const data = await response.json();
				dispatch({ type: "SUCCESS", payload: data });
			} catch (error) {
				const errorObject =
					error instanceof Error ? error : new Error(String(error));
				dispatch({ type: "ERROR", payload: errorObject });
			}
		},
		[initialBody, method, url]
	);

	React.useEffect(() => {
		if (method === "GET") {
			executeRequest();
		}
	}, [method, executeRequest]);

	return { ...state, executeRequest };
};

export default useFetch;
