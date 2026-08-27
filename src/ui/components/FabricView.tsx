/**
 * 3D fabric view for one configuration: the machine's chips drawn on their
 * physical mesh — torus axes as a cube grid with the ICI wiring, switched
 * fabrics as rings around a switch, shaded by pipeline stage. Geometry comes
 * from the resolved Mesh, so the picture matches the placement. Drag to orbit.
 */
import { useEffect, useMemo, useRef } from 'react';
import type { Mesh } from '../../core/engine/surface/deploy';
import { roleSize } from '../../core/engine/surface/deploy';

const CANVAS_W = 375;
const CANVAS_H = 240;
const HALF = 0.39;
const SPACING = 1.18;
const MAX_CHIPS = 260;

type Vec3 = [number, number, number];

/* ------------------------------ colors ------------------------------ */

function cssRGB(styles: CSSStyleDeclaration, name: string): Vec3 {
  const v = styles.getPropertyValue(name).trim();
  const hex6 = /^#([0-9a-f]{6})\b/i.exec(v);
  if (hex6) {
    const n = parseInt(hex6[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const hex3 = /^#([0-9a-f]{3})\b/i.exec(v);
  if (hex3) return [...hex3[1]].map((c) => parseInt(c + c, 16)) as Vec3;
  const rgb = /^rgba?\(\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)/i.exec(v);
  if (rgb) return [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])];
  return [128, 128, 128];
}
const mix = (a: Vec3, b: Vec3, t: number) =>
  a.map((v, i) => Math.round(v + (b[i] - v) * t)) as Vec3;
const rgba = (c: Vec3, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

interface Palette {
  surface: Vec3;
  text: Vec3;
  fabric: Vec3;
  dark: boolean;
}
function palette(): Palette {
  const s = getComputedStyle(document.documentElement);
  const surface = cssRGB(s, '--surface');
  return {
    surface,
    text: cssRGB(s, '--text'),
    fabric: cssRGB(s, '--fabric'),
    dark: 0.2126 * surface[0] + 0.7152 * surface[1] + 0.0722 * surface[2] < 100,
  };
}

// PP-stage tint: the fabric color composited over the surface, deeper per stage.
// Dark surfaces need a higher floor, else stage 0 sits near the background.
function stageColor(pal: Palette, s: number, pp: number): Vec3 {
  const floor = pal.dark ? 0.26 : 0.18;
  const single = pal.dark ? 0.32 : 0.22;
  const span = pal.dark ? 0.5 : 0.6;
  const a = pp === 1 ? single : floor + span * (s / Math.max(1, pp - 1));
  return mix(pal.surface, pal.fabric, a);
}

/* --------------------- mesh -> chip geometry --------------------- */

interface Chip {
  pos: Vec3;
  coord: number[]; // physical coordinate per shown axis
  pp: number; // pipeline stage
  hub?: Vec3; // switched fabric: the node's switch it connects to
}
interface Pod {
  chips: Chip[];
  extent: number[];
  ring: boolean[]; // per physical axis
  wrap: boolean[];
  pp: number;
  switched: boolean;
  hubs: Vec3[]; // one switch per scale-out node
}

function buildPod(mesh: Mesh): Pod {
  const dims = mesh.dims;
  const total = dims.reduce((p, d) => p * d.size, 1);
  const sizeOf = new Map(dims.map((d) => [d.name, d.size]));
  // physical axes actually used, largest first, at most three
  const axisMap = new Map(dims.map((d) => [d.axis.name, d.axis]));
  const axes = [...axisMap.values()].sort((a, b) => b.size - a.size).slice(0, 3);
  const axisIndex = new Map(axes.map((a, i) => [a.name, i]));
  const extent = [1, 1, 1];
  axes.forEach((a, i) => (extent[i] = a.size));
  const pp = roleSize(mesh, 'PP');

  // no ring axes => a switched / all-to-all fabric: chips have no spatial
  // neighbours, so draw each scale-out node's domain as a ring around its
  // switch rather than a grid that would imply a mesh
  const switched = !axes.some((a) => a.kind === 'ring');
  const domI = switched
    ? Math.max(
        0,
        axes.findIndex((a) => a.name === 'D'),
      )
    : 0;
  const domSize = extent[domI];
  const nodeIdx = axes.map((_, i) => i).filter((i) => i !== domI);
  const nodeCount = nodeIdx.reduce((p, i) => p * extent[i], 1);
  const R = Math.max(1.6, domSize * 0.2);
  const step = 2 * R + 2.4;
  const hubs: Vec3[] = Array.from({ length: switched ? nodeCount : 0 }, (_, n) => [
    (n - (nodeCount - 1) / 2) * step,
    0,
    0,
  ]);

  const chips: Chip[] = [];
  for (let idx = 0; idx < total; idx++) {
    const dimCoord: Record<string, number> = {};
    let rem = idx;
    for (const d of dims) {
      dimCoord[d.name] = rem % d.size;
      rem = Math.floor(rem / d.size);
    }
    const coord = [0, 0, 0];
    for (const d of dims) {
      const ai = axisIndex.get(d.axis.name);
      if (ai !== undefined) coord[ai] += dimCoord[d.name] * d.stride;
    }
    let ppRank = 0;
    for (const name of mesh.roles.PP)
      ppRank = ppRank * (sizeOf.get(name) ?? 1) + (dimCoord[name] ?? 0);

    let pos: Vec3;
    let hub: Vec3 | undefined;
    if (switched) {
      const node = nodeIdx.reduce((acc, i) => acc * extent[i] + coord[i], 0);
      hub = hubs[node];
      const angle = (coord[domI] / domSize) * 2 * Math.PI - Math.PI / 2;
      pos = [hub[0] + R * Math.cos(angle), R * Math.sin(angle), 0];
    } else {
      pos = coord.map((c, k) => (c - (extent[k] - 1) / 2) * SPACING) as Vec3;
    }
    chips.push({ coord: coord.slice(0, axes.length), pos, pp: ppRank, hub });
  }
  return {
    chips,
    extent,
    ring: axes.map((a) => a.kind === 'ring'),
    wrap: axes.map((a) => a.wrap),
    pp,
    switched,
    hubs,
  };
}

/* ------------------------------ render ------------------------------ */

const FACES = [
  { axis: 0, sign: 1, idx: [1, 3, 7, 5] },
  { axis: 0, sign: -1, idx: [0, 4, 6, 2] },
  { axis: 1, sign: 1, idx: [2, 6, 7, 3] },
  { axis: 1, sign: -1, idx: [0, 1, 5, 4] },
  { axis: 2, sign: 1, idx: [4, 5, 7, 6] },
  { axis: 2, sign: -1, idx: [0, 2, 3, 1] },
];
function drawScene(
  cv: HTMLCanvasElement | null,
  pod: Pod,
  orbit: { yaw: number; pitch: number },
  pal: Palette,
): void {
  if (!cv) return;
  const ctx = cv.getContext('2d');
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  cv.width = CANVAS_W * dpr;
  cv.height = CANVAS_H * dpr;

  const cy = Math.cos(orbit.yaw),
    sy = Math.sin(orbit.yaw);
  const cp = Math.cos(orbit.pitch),
    sp = Math.sin(orbit.pitch);
  const proj = (x: number, y: number, z: number) => {
    const x1 = x * cy - y * sy;
    const y1 = x * sy + y * cy;
    return { sx: x1, sy: y1 * sp + z * cp, d: y1 * cp - z * sp };
  };
  const axDepth = [proj(1, 0, 0).d, proj(0, 1, 0).d, proj(0, 0, 1).d];

  let horiz = 0.001,
    vert = 0.001;
  for (const ch of pod.chips) {
    const rg = Math.hypot(Math.abs(ch.pos[0]) + HALF, Math.abs(ch.pos[1]) + HALF);
    horiz = Math.max(horiz, rg);
    vert = Math.max(vert, rg * sp + (Math.abs(ch.pos[2]) + HALF) * cp);
  }
  const S = 0.92 * Math.min((CANVAS_W * 0.9) / (2 * horiz), (CANVAS_H * 0.86) / (2 * vert));
  const toScreen = (p: Vec3): [number, number] => {
    const q = proj(p[0], p[1], p[2]);
    return [CANVAS_W / 2 + q.sx * S, CANVAS_H / 2 - q.sy * S];
  };

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

  // ICI wiring: neighbor links per ring axis, plus wrap cables bowed outward
  const byCoord = new Map(pod.chips.map((ch) => [ch.coord.join(','), ch]));
  const wrapCtrl = (pa: Vec3, pb: Vec3, axis: number): Vec3 => {
    const mid = pa.map((v, m) => (v + pb[m]) / 2) as Vec3;
    const d: Vec3 = [...mid];
    d[axis] = 0;
    const len = Math.hypot(d[0], d[1], d[2]);
    const u: Vec3 =
      len > 0.3 ? (d.map((v) => v / len) as Vec3) : axis === 2 ? [0, 1, 0] : [0, 0, 1];
    const bow = 1.1 + 0.12 * pod.extent[axis];
    return mid.map((v, m) => v + u[m] * bow) as Vec3;
  };
  ctx.strokeStyle = rgba(pal.text, 0.14);
  ctx.lineWidth = 1;
  for (const ch of pod.chips)
    ch.coord.forEach((c, ax) => {
      if (!pod.ring[ax]) return;
      if (c + 1 < pod.extent[ax]) {
        const nb = byCoord.get(ch.coord.map((v, i) => (i === ax ? v + 1 : v)).join(','));
        if (nb) {
          const a = toScreen(ch.pos),
            b = toScreen(nb.pos);
          ctx.beginPath();
          ctx.moveTo(a[0], a[1]);
          ctx.lineTo(b[0], b[1]);
          ctx.stroke();
        }
      } else if (pod.wrap[ax] && pod.extent[ax] > 2 && c === pod.extent[ax] - 1) {
        const nb = byCoord.get(ch.coord.map((v, i) => (i === ax ? 0 : v)).join(','));
        if (nb) {
          const a = toScreen(ch.pos),
            b = toScreen(nb.pos),
            cc = toScreen(wrapCtrl(ch.pos, nb.pos, ax));
          ctx.beginPath();
          ctx.moveTo(a[0], a[1]);
          ctx.quadraticCurveTo(cc[0], cc[1], b[0], b[1]);
          ctx.stroke();
        }
      }
    });

  // switched fabric: a switch per scale-out node, spokes to each of its chips
  if (pod.switched) {
    ctx.strokeStyle = rgba(pal.text, 0.1);
    ctx.lineWidth = 1;
    for (const ch of pod.chips) {
      if (!ch.hub) continue;
      const a = toScreen(ch.pos),
        b = toScreen(ch.hub);
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    }
    ctx.fillStyle = rgba(mix(pal.surface, pal.text, 0.35));
    for (const h of pod.hubs) {
      const [x, y] = toScreen(h);
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  // cubes, painter's order (far to near)
  const order = pod.chips
    .map((ch) => ({ ch, d: proj(ch.pos[0], ch.pos[1], ch.pos[2]).d }))
    .sort((a, b) => b.d - a.d);
  for (const { ch } of order) {
    const col = stageColor(pal, ch.pp, pod.pp);
    const corners: [number, number][] = [];
    for (let m = 0; m < 8; m++)
      corners.push(
        toScreen([
          ch.pos[0] + (m & 1 ? HALF : -HALF),
          ch.pos[1] + (m & 2 ? HALF : -HALF),
          ch.pos[2] + (m & 4 ? HALF : -HALF),
        ]),
      );
    for (const f of FACES) {
      if (axDepth[f.axis] * f.sign >= 0) continue; // backface
      const shade = f.axis === 2 ? 1.06 : f.axis === 0 ? 0.9 : 0.8;
      ctx.beginPath();
      ctx.moveTo(corners[f.idx[0]][0], corners[f.idx[0]][1]);
      for (let m = 1; m < 4; m++) ctx.lineTo(corners[f.idx[m]][0], corners[f.idx[m]][1]);
      ctx.closePath();
      ctx.fillStyle = rgba(col.map((v) => Math.min(255, Math.round(v * shade))) as Vec3);
      ctx.fill();
      ctx.strokeStyle = rgba(pal.text, 0.14);
      ctx.lineWidth = 0.75;
      ctx.stroke();
    }
  }
}

/* ------------------------------ component ------------------------------ */

export function FabricView({ mesh }: { mesh: Mesh }) {
  const pod = useMemo(() => buildPod(mesh), [mesh]);
  const orbit = useRef({ yaw: 0.7, pitch: 0.5, drag: false, touched: false, px: 0, py: 0 });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // theme colors change only when data-theme flips (watched below), so read them
  // once instead of on every animation frame
  const palRef = useRef<Palette | null>(null);
  const tooBig = pod.chips.length > MAX_CHIPS;

  useEffect(() => {
    if (tooBig) return;
    // a new config gets a fresh camera and resumes the idle spin, even if the
    // previous one was grabbed (this effect only re-runs when the pod changes)
    Object.assign(orbit.current, { yaw: 0.7, pitch: 0.5, drag: false, touched: false });
    const draw = () => drawScene(canvasRef.current, pod, orbit.current, palRef.current!);
    palRef.current = palette();
    draw();
    const mo = new MutationObserver(() => {
      palRef.current = palette();
      draw();
    });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    let raf = 0;
    let last = performance.now();
    const tick = (t: number) => {
      if (orbit.current.touched) return;
      orbit.current.yaw += (t - last) * 0.00012;
      last = t;
      draw();
      raf = requestAnimationFrame(tick);
    };
    if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches)
      raf = requestAnimationFrame(tick);
    return () => {
      mo.disconnect();
      cancelAnimationFrame(raf);
    };
  }, [pod, tooBig]);

  if (tooBig)
    return <div className="fabric-note">{pod.chips.length} chips — too many to draw legibly.</div>;

  const redraw = () =>
    palRef.current && drawScene(canvasRef.current, pod, orbit.current, palRef.current);
  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    Object.assign(orbit.current, { drag: true, touched: true, px: e.clientX, py: e.clientY });
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const o = orbit.current;
    if (!o.drag) return;
    o.yaw += (e.clientX - o.px) * 0.008;
    o.pitch = Math.max(0.05, Math.min(1.35, o.pitch + (e.clientY - o.py) * 0.006));
    o.px = e.clientX;
    o.py = e.clientY;
    redraw();
  };
  const onUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    orbit.current.drag = false;
    e.currentTarget.releasePointerCapture(e.pointerId);
  };

  return (
    <div className="fabric">
      <canvas
        ref={canvasRef}
        className="fabric-canvas"
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        title="drag to orbit"
      />
    </div>
  );
}
