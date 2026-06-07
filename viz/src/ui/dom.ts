// Tiny DOM helpers for building controls and metric readouts.

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  ...children: Array<Node | string>
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(typeof c === "string" ? document.createTextNode(c) : c);
  return node;
}

export function button(label: string, onClick: () => void, primary = false): HTMLButtonElement {
  const b = el("button", primary ? { class: "primary" } : {}, label);
  b.addEventListener("click", onClick);
  return b;
}

export interface SliderHandle {
  el: HTMLElement;
  get(): number;
  set(v: number): void;
}

export function slider(
  label: string,
  min: number,
  max: number,
  stepSize: number,
  value: number,
  onInput: (v: number) => void,
  format: (v: number) => string = (v) => String(v),
): SliderHandle {
  const valSpan = el("span", { class: "val" }, format(value));
  const input = el("input", {
    type: "range",
    min: String(min),
    max: String(max),
    step: String(stepSize),
    value: String(value),
  });
  input.addEventListener("input", () => {
    const v = parseFloat(input.value);
    valSpan.textContent = format(v);
    onInput(v);
  });
  const wrap = el("label", { class: "field" }, el("span", {}, `${label} `, valSpan), input);
  return {
    el: wrap,
    get: () => parseFloat(input.value),
    set: (v: number) => {
      input.value = String(v);
      valSpan.textContent = format(v);
    },
  };
}

export function selectControl(
  label: string,
  options: Array<{ value: string; label: string }>,
  value: string,
  onChange: (v: string) => void,
): HTMLElement {
  const sel = el("select", {});
  for (const o of options) {
    const opt = el("option", { value: o.value }, o.label);
    if (o.value === value) opt.setAttribute("selected", "selected");
    sel.append(opt);
  }
  sel.addEventListener("change", () => onChange(sel.value));
  return el("label", { class: "field" }, el("span", {}, label), sel);
}

export interface MetricHandle {
  el: HTMLElement;
  set(v: string, cls?: string): void;
}

export function metric(key: string, accent?: "temporal" | "structural"): MetricHandle {
  const v = el("span", { class: `v${accent ? " " + accent : ""}` }, "—");
  const wrap = el("div", { class: "metric" }, el("span", { class: "k" }, key), v);
  return {
    el: wrap,
    set: (text: string) => {
      v.textContent = text;
    },
  };
}
