import { useEffect, useState } from "react";

import { api } from "../../lib/api";
import type { ChatMessage } from "../../types";

/** Data URLs are large; keep one copy per stored asset for the session. */
const cache = new Map<string, string>();

interface ChatImageProps {
  image: NonNullable<ChatMessage["images"]>[number];
  className?: string;
  alt?: string;
}

/**
 * Images attached before the disk-store migration still carry a data URL.
 * Reloaded chats carry an asset reference instead, which is read lazily so a
 * shelf full of screenshots does not inflate startup.
 */
export function ChatImage({ image, className, alt }: ChatImageProps) {
  const [src, setSrc] = useState<string | undefined>(() =>
    image.data_url ?? (image.asset ? cache.get(image.asset) : undefined),
  );
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (src || !image.asset) return;
    let cancelled = false;
    void api
      .chatAssetDataUrl(image.asset)
      .then((dataUrl) => {
        if (cancelled) return;
        cache.set(image.asset!, dataUrl);
        setSrc(dataUrl);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [image.asset, src]);

  if (failed) {
    return (
      <span
        className={`flex items-center justify-center bg-white/5 text-[8px] text-slate-600 ${className ?? ""}`}
      >
        image unavailable
      </span>
    );
  }
  if (!src) {
    return (
      <span
        className={`animate-pulse bg-white/5 ${className ?? ""}`}
        aria-label={alt ?? image.name}
      />
    );
  }
  return <img src={src} alt={alt ?? image.name} className={className} />;
}
