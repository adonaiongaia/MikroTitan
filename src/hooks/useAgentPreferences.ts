import { useCallback } from "react";

import { usePersistentState } from "./usePersistentState";
import type { AgentPreferences } from "../types";

export const DEFAULT_AGENT_PREFERENCES: AgentPreferences = {
  enabled: false,
  maxIterations: 10,
  approvalPolicy: "destructive",
  workspaceRoot: "",
  planningStep: false,
  autoCompact: true,
  compactionThreshold: 0.75,
};

/** Agent configuration is machine-local, like the sampling preferences. */
export function useAgentPreferences() {
  const [preferences, setPreferences] = usePersistentState<AgentPreferences>(
    "janas.agentPreferences.v1",
    DEFAULT_AGENT_PREFERENCES,
  );

  const update = useCallback(
    <K extends keyof AgentPreferences>(key: K, value: AgentPreferences[K]) => {
      setPreferences((previous) => ({ ...previous, [key]: value }));
    },
    [setPreferences],
  );

  return { preferences, update, setPreferences };
}
