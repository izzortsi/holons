// Canvas rendering: scalar fields (viridis) and Game-of-Life grids with
// optional per-component tinting. Renders into an n-by-n offscreen buffer then
// scales up with nearest-neighbour for crisp pixels; vector overlays (centroids,
// predicted anchors) draw on top in display coordinates.

// Viridis colour stops (0..1), interpolated into a 256-entry LUT.
const VIRIDIS_STOPS: Array<[number, number, number]> = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
  [31, 158, 137], [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37],
];

function buildViridis(): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (VIRIDIS_STOPS.length - 1);
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, VIRIDIS_STOPS.length - 1);
    const f = t - lo;
    for (let k = 0; k < 3; k++) {
      lut[i * 3 + k] = VIRIDIS_STOPS[lo][k] * (1 - f) + VIRIDIS_STOPS[hi][k] * f;
    }
  }
  return lut;
}
const VIRIDIS = buildViridis();

// Distinct palette for Game-of-Life components.
const COMPONENT_PALETTE: Array<[number, number, number]> = [
  [240, 180, 41], [59, 130, 246], [63, 185, 80], [229, 72, 77],
  [168, 85, 247], [45, 212, 191], [244, 114, 182], [250, 204, 21],
];

export class GridCanvas {
  readonly canvas: HTMLCanvasElement;
  readonly ctx: CanvasRenderingContext2D;
  readonly scale: number;
  private off: HTMLCanvasElement;
  private octx: CanvasRenderingContext2D;
  private img: ImageData;

  constructor(n: number, displaySize: number) {
    this.canvas = document.createElement("canvas");
    this.canvas.width = displaySize;
    this.canvas.height = displaySize;
    this.scale = displaySize / n;
    this.ctx = this.canvas.getContext("2d")!;
    this.ctx.imageSmoothingEnabled = false;
    this.off = document.createElement("canvas");
    this.off.width = n;
    this.off.height = n;
    this.octx = this.off.getContext("2d")!;
    this.img = this.octx.createImageData(n, n);
  }

  private blit(): void {
    this.octx.putImageData(this.img, 0, 0);
    this.ctx.drawImage(
      this.off, 0, 0, this.img.width, this.img.height,
      0, 0, this.canvas.width, this.canvas.height,
    );
  }

  // Draw a scalar field in [0,1] with viridis.
  drawScalar(field: Float32Array, vmax = 1): void {
    const d = this.img.data;
    for (let i = 0; i < field.length; i++) {
      let v = field[i] / vmax;
      if (v < 0) v = 0;
      else if (v > 1) v = 1;
      const c = (Math.round(v * 255) * 3) | 0;
      d[i * 4] = VIRIDIS[c];
      d[i * 4 + 1] = VIRIDIS[c + 1];
      d[i * 4 + 2] = VIRIDIS[c + 2];
      d[i * 4 + 3] = 255;
    }
    this.blit();
  }

  // Draw a binary GoL field; if compIndex is given, tint each component.
  drawLife(cells: Uint8Array, compIndex?: Int32Array): void {
    const d = this.img.data;
    for (let i = 0; i < cells.length; i++) {
      let r = 14, g = 17, b = 22;
      if (cells[i]) {
        if (compIndex && compIndex[i] >= 0) {
          const p = COMPONENT_PALETTE[compIndex[i] % COMPONENT_PALETTE.length];
          [r, g, b] = p;
        } else {
          r = 214; g = 221; b = 230;
        }
      }
      d[i * 4] = r;
      d[i * 4 + 1] = g;
      d[i * 4 + 2] = b;
      d[i * 4 + 3] = 255;
    }
    this.blit();
  }

  // Draw a filled circle in grid coordinates.
  marker(row: number, col: number, color: string, radius = 4): void {
    this.ctx.beginPath();
    this.ctx.arc((col + 0.5) * this.scale, (row + 0.5) * this.scale, radius, 0, 2 * Math.PI);
    this.ctx.fillStyle = color;
    this.ctx.fill();
  }

  // Draw a hollow ring in grid coordinates (for predicted/ghost positions).
  ring(row: number, col: number, color: string, radius = 5): void {
    this.ctx.beginPath();
    this.ctx.arc((col + 0.5) * this.scale, (row + 0.5) * this.scale, radius, 0, 2 * Math.PI);
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 2;
    this.ctx.stroke();
  }
}

export function componentColor(i: number): string {
  const p = COMPONENT_PALETTE[i % COMPONENT_PALETTE.length];
  return `rgb(${p[0]},${p[1]},${p[2]})`;
}
