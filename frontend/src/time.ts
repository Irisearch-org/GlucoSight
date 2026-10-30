export function minutesAgo(n: number): Date {
  return new Date(Date.now() - n * 60_000);
}

/** Value for <input type="datetime-local">: local time, no timezone suffix. */
export function toInputValue(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** t0_timestamp / fingerstick_timestamp sent to the API: full ISO8601. */
export function toApiTimestamp(d: Date): string {
  return d.toISOString();
}

export function ageMinutes(time: string): number {
  return Math.max(0, Math.round((Date.now() - new Date(time).getTime()) / 60_000));
}

export function describeAge(minutes: number): string {
  if (minutes < 1) return "Baru saja";
  if (minutes < 60) return `${minutes} menit yang lalu`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} jam yang lalu`;
  return `${Math.round(hours / 24)} hari yang lalu`;
}

const dateFmt = new Intl.DateTimeFormat("id-ID", { day: "numeric", month: "short", year: "numeric" });
const timeFmt = new Intl.DateTimeFormat("id-ID", { hour: "2-digit", minute: "2-digit" });

export function formatWhen(iso: string): string {
  const d = new Date(iso);
  const sameDay = d.toDateString() === new Date().toDateString();
  return sameDay ? `Hari ini, ${timeFmt.format(d)}` : `${dateFmt.format(d)}, ${timeFmt.format(d)}`;
}
