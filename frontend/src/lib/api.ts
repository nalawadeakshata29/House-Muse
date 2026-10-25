import axios from "axios";

export type User = { id: number; username: string; email: string; first_name?: string; last_name?: string; auth_provider?: string; avatar_url?: string };
export type AuthResponse = { user: User; token: string };
export type RoomType = "living_room" | "bedroom" | "kitchen" | "dining_room" | "bathroom" | "study_room";
export type Scope = "home_decor" | "interior_design" | "full_makeover";
export type StyleId = "vintage_indian" | "contemporary_indian" | "coastal" | "industrial" | "scandinavian" | "eclectic" | "shabby_chic_style" | "mid_century_modern" | "victorian";
export type Project = { id: number; name: string; room_type: string; length: number; width: number; height: number; styles: string[]; budget: number; scope: string; source_image?: string; generated_image_url?: string; result?: { budget_breakdown?: Record<string, number>; over_budget?: boolean; shortfall_amount?: number; total_estimated?: number; budget_remaining?: number }; created_at: string };

// VITE_API_BASE may be set with or without the trailing "/api" (e.g. "http://localhost:8000").
// Normalise so every request lands on the Django "/api/..." routes either way.
const RAW_API_BASE = ((import.meta.env.VITE_API_BASE as string | undefined) || "http://localhost:8000").trim().replace(/\/+$/, "");
export const API_BASE = /\/api$/.test(RAW_API_BASE) ? RAW_API_BASE : `${RAW_API_BASE}/api`;

export const api = axios.create({ baseURL: API_BASE });

// Public endpoints must NOT receive a token. A stale token on these calls makes
// Knox reject the request with "Invalid token" before the login/register view runs.
const PUBLIC_PATHS = [
  "/auth/login/",
  "/auth/register/",
  "/auth/google/",
  "/auth/password-reset/",
  "/auth/password-reset-confirm/",
];

api.interceptors.request.use(config => {
  const isPublic = PUBLIC_PATHS.some(path => config.url?.includes(path));
  const token = localStorage.getItem("house_muse_token");
  if (token && !isPublic) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// If the server says the token is bad, clear the saved session so the user
// isn't stuck with a stale token after a database change or token expiry.
api.interceptors.response.use(
  response => response,
  error => {
    if (error?.response?.status === 401) {
      localStorage.removeItem("house_muse_token");
      localStorage.removeItem("house_muse_user");
    }
    return Promise.reject(error);
  }
);

export const saveSession = (data: AuthResponse) => {
  localStorage.setItem("house_muse_token", data.token);
  localStorage.setItem("house_muse_user", JSON.stringify(data.user));
};

export const login = async (username: string, password: string) => {
  const { data } = await api.post<AuthResponse>("/auth/login/", { username, password });
  saveSession(data);
  return data;
};

export const register = async (payload: { username: string; email: string; password: string; first_name?: string; last_name?: string }) => {
  const { data } = await api.post<AuthResponse>("/auth/register/", payload);
  saveSession(data);
  return data;
};

export const googleSignIn = async (id_token: string) => {
  const { data } = await api.post<AuthResponse>("/auth/google/", { id_token });
  saveSession(data);
  return data;
};

export const getMe = async () => (await api.get<User>("/auth/me/")).data;

export const logout = async () => {
  await api.post("/auth/logout/");
  localStorage.removeItem("house_muse_token");
  localStorage.removeItem("house_muse_user");
};

export const requestPasswordReset = async (email: string) => (await api.post("/auth/password-reset/", { email })).data;

export const confirmPasswordReset = async (uid: string, token: string, new_password: string) =>
  (await api.post("/auth/password-reset-confirm/", { uid, token, new_password })).data;

export const getProjects = async () => (await api.get<Project[]>("/studio/projects/")).data;

export const analyse = async (payload: { room_type: RoomType; length: number; width: number; height: number; styles: StyleId[]; budget: number; scope: Scope; num_rooms?: number; harmony?: string; save?: boolean; image?: File }) => {
  const body = new FormData();
  body.append("room_type", payload.room_type);
  body.append("length", String(payload.length));
  body.append("width", String(payload.width));
  body.append("height", String(payload.height));
  payload.styles.forEach(style => body.append("styles", style));
  body.append("budget", String(payload.budget));
  body.append("scope", payload.scope);
  body.append("num_rooms", String(payload.num_rooms || 1));
  body.append("harmony", payload.harmony || "analogous");
  body.append("save", String(payload.save ?? true));
  if (payload.image) body.append("image", payload.image);
  return (await api.post("/studio/analyse/", body)).data;
};

export const generateImage = async (payload: Record<string, unknown>) => (await api.post("/studio/generate-image/", payload)).data;

export type RefineHistoryItem = {
  id: number;
  instruction: string;
  resulting_image: string;
  created_at: string;
  image_id?: string;
};

export type RefineResponse = {
  success: boolean;
  image_id?: string;
  image_url: string;
  image_b64?: string;
  resulting_image: string;
  instruction: string;
  history: RefineHistoryItem[];
  project_id: number;
  error?: string;
};

export const refineImage = async (payload: {
  project_id: number;
  instruction: string;
  image_id?: string;
  current_image_id?: string;
  init_image?: string;
}): Promise<RefineResponse> =>
  (await api.post<RefineResponse>("/studio/refine-image/", payload)).data;

export function isImageExpiredError(error: unknown): boolean {
  const resp = (error as { response?: { status?: number; data?: Record<string, unknown> } })?.response;
  return resp?.status === 404 && (resp?.data?.code === "image_expired" || resp?.data?.error === "image_expired");
}

export function apiError(error: unknown) {
  const response = (error as { response?: { data?: Record<string, unknown> | string } })?.response?.data;
  if (!response) return "We could not reach the House Muse service. Check VITE_API_BASE and your backend CORS settings.";
  if (typeof response === "string") return response;
  if (typeof response === "object") {
    if (response.error && typeof response.error === "string") return response.error;
    if (response.detail && typeof response.detail === "string") return response.detail;
    const first = Object.entries(response)[0];
    if (first) {
      const val = Array.isArray(first[1]) ? first[1][0] : first[1];
      return `${first[0]}: ${val}`;
    }
  }
  return "The request could not be completed.";
}