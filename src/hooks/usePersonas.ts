import { useCallback } from "react";

import { usePersistentState } from "./usePersistentState";
import type { Persona } from "../types";

function personaId(): string {
  return `persona-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 8)}`;
}

function isPersona(value: unknown): value is Persona {
  if (!value || typeof value !== "object") return false;
  const persona = value as Partial<Persona>;
  return (
    typeof persona.id === "string" &&
    typeof persona.name === "string" &&
    typeof persona.prompt === "string" &&
    persona.prompt.trim().length > 0
  );
}

function sanitize(value: Persona[]): Persona[] {
  return Array.isArray(value) ? value.filter(isPersona) : [];
}

/** Saved system prompts live with the other UI preferences, never on a server. */
export function usePersonas() {
  const [personas, setPersonas] = usePersistentState<Persona[]>(
    "janas.personas.v1",
    [],
  );

  const add = useCallback(
    (name: string, prompt: string): Persona | null => {
      const trimmedPrompt = prompt.trim();
      if (!trimmedPrompt) return null;
      const persona: Persona = {
        id: personaId(),
        name: name.trim() || "Untitled persona",
        prompt: trimmedPrompt,
      };
      setPersonas((previous) => [...sanitize(previous), persona]);
      return persona;
    },
    [setPersonas],
  );

  const remove = useCallback(
    (id: string) => {
      setPersonas((previous) =>
        sanitize(previous).filter((persona) => persona.id !== id),
      );
    },
    [setPersonas],
  );

  return { personas: sanitize(personas), add, remove };
}
