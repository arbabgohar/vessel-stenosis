import sys
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sys.dont_write_bytecode = True

from measure_case import ROOT, prepare_case_path
from pick_points import populate_slice_figure
from preview_case import make_figure
from volume_io import load_nifti_pair


class CaseWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.job = None
        self.case = None
        self.selection = None
        self.endpoints = [None, None]
        self.preview = None
        self.slice_sliders = []

        root.title("Vessel Stenosis Measurement")
        root.geometry("1250x810")
        root.minsize(850, 650)
        root.protocol("WM_DELETE_WINDOW", self.close)

        layout = ttk.Frame(root, padding=14)
        layout.pack(fill="both", expand=True)
        layout.columnconfigure(0, weight=1)
        layout.rowconfigure(3, weight=1)

        ttk.Label(layout, text="Vessel Stenosis Measurement", font=("Segoe UI", 19, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )
        inputs = ttk.LabelFrame(layout, text="Case files", padding=8)
        inputs.grid(row=1, column=0, sticky="ew")
        inputs.columnconfigure(1, weight=1)
        self.image_file, self.mask_file = tk.StringVar(), tk.StringVar()
        self.labels = tk.StringVar()
        self.input_widgets = []
        for row, (title, variable) in enumerate((("CT image", self.image_file), ("Label mask", self.mask_file))):
            ttk.Label(inputs, text=title).grid(row=row, column=0, sticky="w", padx=(0, 10))
            entry = ttk.Entry(inputs, textvariable=variable)
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            browse = ttk.Button(inputs, text="Browse…", command=lambda target=variable: self.browse(target))
            browse.grid(row=row, column=2, padx=(6, 0))
            self.input_widgets.extend((entry, browse))
        ttk.Label(inputs, text="Vessel labels").grid(row=2, column=0, sticky="w")
        labels_field = ttk.Entry(inputs, textvariable=self.labels, width=18)
        labels_field.grid(row=2, column=1, sticky="w", pady=3)
        self.input_widgets.append(labels_field)
        self.load_button = ttk.Button(inputs, text="Load case", command=self.load_case)
        self.load_button.grid(row=2, column=2, sticky="ew", padx=(6, 0))

        actions = ttk.Frame(layout)
        actions.grid(row=2, column=0, sticky="ew", pady=10)
        self.endpoint_buttons = []
        self.endpoint_labels = [tk.StringVar(value="—"), tk.StringVar(value="—")]
        for index, name in enumerate(("Start", "End")):
            button = ttk.Button(actions, text=f"Set {name.lower()}", command=lambda slot=index: self.set_endpoint(slot))
            button.pack(side="left")
            ttk.Label(actions, textvariable=self.endpoint_labels[index], width=18).pack(side="left", padx=(6, 12))
            self.endpoint_buttons.append(button)
        self.measure_button = ttk.Button(actions, text="Measure and preview", command=self.measure)
        self.measure_button.pack(side="left")
        self.save_button = ttk.Button(actions, text="Save PNG…", command=self.save_preview)
        self.save_button.pack(side="right")

        self.tabs = ttk.Notebook(layout)
        self.tabs.grid(row=3, column=0, sticky="nsew")
        self.slices_tab, self.preview_tab = ttk.Frame(self.tabs), ttk.Frame(self.tabs)
        self.tabs.add(self.slices_tab, text="Choose endpoints")
        self.tabs.add(self.preview_tab, text="Measurement preview")
        ttk.Label(self.slices_tab, text="Choose a CT image and label mask, then load the case.").place(
            relx=0.5, rely=0.5, anchor="center"
        )
        self.status = tk.StringVar(value="Vessel labels can be entered as numbers separated by spaces.")
        ttk.Label(layout, textvariable=self.status, wraplength=790).grid(row=4, column=0, sticky="w", pady=(8, 0))
        for variable in (self.image_file, self.mask_file, self.labels):
            variable.trace_add("write", self.invalidate_case)
        self.update_buttons()

    def browse(self, variable):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("NIfTI volumes", "*.nii *.nii.gz")])
        if path:
            variable.set(path)

    def invalidate_case(self, *_):
        if self.case is not None:
            self.case = self.selection = self.preview = None
            self.endpoints = [None, None]
            self.slice_sliders = []
            for label in self.endpoint_labels:
                label.set("—")
            for tab in (self.slices_tab, self.preview_tab):
                for widget in tab.winfo_children():
                    widget.destroy()
            self.tabs.select(self.slices_tab)
            self.status.set("Load the case to apply the new inputs.")
        self.update_buttons()

    def update_buttons(self):
        idle = self.job is None
        for widget in self.input_widgets:
            widget.configure(state="normal" if idle else "disabled")
        ready_to_load = idle and all(
            field.get().strip() for field in (self.image_file, self.mask_file, self.labels)
        )
        self.load_button.configure(state="normal" if ready_to_load else "disabled")
        for button in self.endpoint_buttons:
            button.configure(state="normal" if idle and self.selection is not None else "disabled")
        ready_to_measure = idle and all(point is not None for point in self.endpoints)
        self.measure_button.configure(state="normal" if ready_to_measure else "disabled")
        self.save_button.configure(state="normal" if idle and self.preview is not None else "disabled")

    def submit(self, status, operation, finished, *args):
        self.status.set(status)
        self.job = self.executor.submit(operation, *args)
        self.update_buttons()
        self.root.after(60, self.poll_job, finished)

    def poll_job(self, finished):
        if not self.job.done():
            self.root.after(60, self.poll_job, finished)
            return
        job, self.job = self.job, None
        try:
            finished(job.result())
        except Exception as exc:
            self.status.set(str(exc))
            messagebox.showerror("Vessel measurement", str(exc), parent=self.root)
        finally:
            self.update_buttons()

    def load_case(self):
        try:
            labels = [float(value) for value in self.labels.get().split()]
        except ValueError:
            messagebox.showerror("Vessel labels", "Enter numbers separated by spaces.", parent=self.root)
            return
        self.invalidate_case()
        self.submit("Loading CT and label mask…", load_nifti_pair, self.show_case,
                    Path(self.image_file.get().strip()), Path(self.mask_file.get().strip()), labels)

    def show_case(self, case):
        self.case = case
        ct, mask, spacing, labels = case
        for widget in self.slices_tab.winfo_children():
            widget.destroy()
        figure = Figure(figsize=(12, 5))
        canvas = FigureCanvasTkAgg(figure, master=self.slices_tab)
        self.slice_sliders = populate_slice_figure(figure, ct, mask, spacing, self.select_voxel)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.tabs.select(self.slices_tab)
        self.status.set(f"Labels found: {labels.tolist()}. Select two voxels inside the red mask.")

    def select_voxel(self, point, inside):
        self.selection = point if inside else None
        self.update_buttons()

    def set_endpoint(self, index):
        self.endpoints[index] = self.selection
        self.endpoint_labels[index].set(str(self.selection))
        self.preview = None
        for widget in self.preview_tab.winfo_children():
            widget.destroy()
        self.update_buttons()

    def measure(self):
        if self.endpoints[0] == self.endpoints[1]:
            self.status.set("Choose two different voxels inside the vessel mask.")
            return
        ct, mask, spacing, _ = self.case
        target_spacing = 0.4
        self.submit("Finding the vessel path…", prepare_case_path,
                    lambda output: self.show_preview(output, target_spacing),
                    ct, mask, spacing, *self.endpoints, target_spacing)

    def show_preview(self, output, spacing):
        mask, path = output
        figure, result = make_figure(mask, path, spacing)
        for widget in self.preview_tab.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(figure, master=self.preview_tab)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        self.preview = figure
        self.tabs.select(self.preview_tab)
        self.status.set(
            f"Path {result.distance_mm[-1]:.2f} mm · Minimum {result.min_diameter_mm:.2f} mm · "
            f"Reference {result.reference_diameter_mm:.2f} mm · Narrowing {result.percent_stenosis:.1f}%"
        )

    def save_preview(self):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".png",
                                            initialfile="vessel_preview.png", filetypes=[("PNG figure", "*.png")])
        if not path:
            return
        try:
            output = Path(path).resolve()
            if output.is_relative_to(ROOT):
                raise ValueError("Choose a folder outside the repository.")
            self.preview.savefig(output, dpi=160, format="png")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Save preview", str(exc), parent=self.root)
            return
        self.status.set(f"Saved {output}")

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    CaseWindow(root)
    root.mainloop()
