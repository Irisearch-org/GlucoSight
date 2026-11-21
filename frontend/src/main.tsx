/// <reference types="vite/client" />
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import DishGatePrototype from "./prototype/DishGatePrototype"; // PROTOTYPE — wayfinder #19, throwaway
import "./index.css";
import "./desktop.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {import.meta.env.DEV && new URLSearchParams(window.location.search).get("prototype") === "dish-gate" ? <DishGatePrototype /> : <App />}
  </StrictMode>,
);
