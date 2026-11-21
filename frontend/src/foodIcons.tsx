import { Icon } from "./ui";

/** The dish table has thousands of rows, so every food gets the same plate icon. */
export function FoodIcon({ size = 40 }: { size?: number }) {
  return (
    <span className="food-icon" style={{ width: size, height: size }}>
      <Icon name="gallery" size={size * 0.5} />
    </span>
  );
}
