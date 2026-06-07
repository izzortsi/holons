// A small live line-plot on a canvas: several named series sharing an x axis,
// autoscaled y, optional fixed y-max, with a legend.

export interface SeriesDef {
  key: string;
  color: string;
  label: string;
}

interface Point {
  x: number;
  values: Record<string, number>;
}

export class LinePlot {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private defs: SeriesDef[] = [];
  private points: Point[] = [];
  private fixedYMax: number | null;
  private xLabel: string;

  constructor(width: number, height: number, opts: { fixedYMax?: number; xLabel?: string } = {}) {
    this.canvas = document.createElement("canvas");
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx = this.canvas.getContext("2d")!;
    this.fixedYMax = opts.fixedYMax ?? null;
    this.xLabel = opts.xLabel ?? "tau";
  }

  setSeries(defs: SeriesDef[]): void {
    this.defs = defs;
  }

  reset(): void {
    this.points = [];
    this.render();
  }

  push(x: number, values: Record<string, number>): void {
    this.points.push({ x, values });
  }

  render(): void {
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;
    const padL = 38, padR = 10, padT = 10, padB = 22;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0e1116";
    ctx.fillRect(0, 0, W, H);

    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    let xMax = 1;
    let yMax = this.fixedYMax ?? 0.001;
    for (const p of this.points) {
      if (p.x > xMax) xMax = p.x;
      if (this.fixedYMax == null) {
        for (const d of this.defs) {
          const v = p.values[d.key];
          if (v != null && v > yMax) yMax = v;
        }
      }
    }
    yMax *= this.fixedYMax == null ? 1.15 : 1;

    const xOf = (x: number) => padL + (x / xMax) * plotW;
    const yOf = (y: number) => padT + plotH - (y / yMax) * plotH;

    // axes
    ctx.strokeStyle = "#2a3140";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();
    ctx.fillStyle = "#8b96a5";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(yMax.toFixed(2), 2, padT + 8);
    ctx.fillText("0", 2, padT + plotH);
    ctx.fillText(this.xLabel, padL + plotW - 14, H - 6);

    // series
    for (const d of this.defs) {
      ctx.strokeStyle = d.color;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      let started = false;
      for (const p of this.points) {
        const v = p.values[d.key];
        if (v == null) continue;
        const px = xOf(p.x);
        const py = yOf(v);
        if (!started) {
          ctx.moveTo(px, py);
          started = true;
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();
    }

    // legend
    let lx = padL + 6;
    for (const d of this.defs) {
      ctx.fillStyle = d.color;
      ctx.fillRect(lx, padT + 2, 9, 9);
      ctx.fillStyle = "#8b96a5";
      ctx.fillText(d.label, lx + 13, padT + 10);
      lx += 16 + d.label.length * 6.2;
    }
  }
}
