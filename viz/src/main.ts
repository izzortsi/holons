import "./style.css";
import { el, button } from "./ui/dom";
import { createGolView } from "./golView";
import { createLeniaView } from "./leniaView";

const app = document.querySelector<HTMLDivElement>("#app")!;

const golView = createGolView();
const leniaView = createLeniaView();

const content = el("div", {}, golView);

const golTab = button("Game of Life", () => activate("gol"));
const leniaTab = button("Lenia", () => activate("lenia"));
golTab.className = "tab";
leniaTab.className = "tab";

function activate(which: "gol" | "lenia"): void {
  golTab.classList.toggle("active", which === "gol");
  leniaTab.classList.toggle("active", which === "lenia");
  content.replaceChildren(which === "gol" ? golView : leniaView);
}

app.append(
  el("h1", {}, "Holons Explorer"),
  el("p", { class: "subtitle" },
    "Live temporal & structural emergence defects on known-rule substrates. "
    + "Watch how each defect is measured — coarse objects, predicted vs actual, "
    + "and the field-level superposition defect."),
  el("div", { class: "tabs" }, golTab, leniaTab),
  content,
);

activate("gol");
