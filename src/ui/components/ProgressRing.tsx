export function ProgressRing({
  frac,
  size = 22,
  color,
}: {
  frac: number;
  size?: number;
  color?: string;
}) {
  const r = size / 2 - 3;
  const c = 2 * Math.PI * r;
  return (
    <svg className="progress-ring" width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle className="ring-bg" cx={size / 2} cy={size / 2} r={r} />
      <circle
        className="ring-fg"
        cx={size / 2}
        cy={size / 2}
        r={r}
        strokeDasharray={c}
        strokeDashoffset={c * (1 - Math.min(1, frac))}
        style={color ? { stroke: color } : undefined}
      />
    </svg>
  );
}
