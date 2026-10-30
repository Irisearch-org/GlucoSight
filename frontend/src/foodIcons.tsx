import { Icon } from "./ui";

const EMOJI: Record<string, string> = {
  bakso: "🍜", "bebek betutu": "🍗", "gado gado": "🥗", gudeg: "🍛",
  "nasi goreng": "🍚", pempek: "🐟", rawon: "🍲", rendang: "🥘", sate: "🍢", soto: "🍲",
};

/** Emoji for the 10 known TKPI dishes; a generic plate icon otherwise. */
export function FoodIcon({ name, size = 40 }: { name: string; size?: number }) {
  const emoji = EMOJI[name.trim().toLowerCase()];
  return (
    <span className="food-icon" style={{ width: size, height: size, fontSize: size * 0.55 }}>
      {emoji ?? <Icon name="gallery" size={size * 0.5} />}
    </span>
  );
}
