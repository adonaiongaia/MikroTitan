import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import { applyTheme, readTheme } from "./lib/theme";
import "./index.css";

// Resolve the saved theme before the first React paint so the window does not
// flash the default palette on launch.
applyTheme(readTheme(), false);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
