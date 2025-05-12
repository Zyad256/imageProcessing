    from tkinter import *
    from PIL import Image, ImageTk, ImageOps, ImageFilter
    from tkinter import filedialog
    import numpy as np

    window = Tk()
    window.geometry("800x750")
    window.resizable(False, False)
    # window.config(bg="#00FF00")
    window.title("Image Filters")
    img = Image.open("icon.jpeg")
    icon = ImageTk.PhotoImage(img)
    window.iconphoto(True, icon)

    IMAGE_DISPLAY_SIZE = (350, 350)
    img = None
    filtered_img = None

    start_x = start_y = end_x = end_y = None
    selection_rect = None

    def on_mouse_down(event):
        global start_x, start_y, selection_rect
        start_x, start_y = event.x, event.y
        if selection_rect:
            canvas.delete(selection_rect)
            selection_rect = None

    def on_mouse_drag(event):
        global end_x, end_y, selection_rect
        end_x, end_y = event.x, event.y
        if selection_rect:
            canvas.delete(selection_rect)
        selection_rect = canvas.create_rectangle(start_x, start_y, end_x, end_y, outline="red", width=2)

    def show_image(image, panel):
        global filtered_img
        image = image.copy().resize(IMAGE_DISPLAY_SIZE, Image.Resampling.LANCZOS)
        img_display = ImageTk.PhotoImage(image)
        if isinstance(panel, Canvas):
            panel.delete("all")
            panel.create_image(0, 0, anchor=NW, image=img_display)
            panel.image = img_display
        else:
            panel.config(image=img_display)
            panel.image = img_display
        if panel == panel_filtered:
            filtered_img = image

    def open_image():
        global img
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                img = Image.open(file_path)
                show_image(img, canvas)
                update_status("Image loaded.")
            except Exception as e:
                update_status(f"Failed to open image: {e}")

    def save_output():
        if filtered_img:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
            if file_path:
                try:
                    filtered_img.save(file_path)
                    update_status(f"Saved to {file_path}")
                except Exception as e:
                    update_status(f"Failed to save image: {e}")

    def blue_filter():
        if img:
            r, g, b = img.convert("RGB").split()
            zero = r.point(lambda _: 0)
            show_image(Image.merge("RGB", (zero, zero, b)), panel_filtered)
            update_status("Blue filter applied.")

    def red_filter():
        if img:
            r, g, b = img.convert("RGB").split()
            zero = r.point(lambda _: 0)
            show_image(Image.merge("RGB", (r, zero, zero)), panel_filtered)
            update_status("Red filter applied.")

    def green_filter():
        if img:
            r, g, b = img.convert("RGB").split()
            zero = r.point(lambda _: 0)
            show_image(Image.merge("RGB", (zero, g, zero)), panel_filtered)
            update_status("Green filter applied.")

    def gray_filter():
        if img:
            gray_img = img.convert("L")
            show_image(gray_img, panel_filtered)
            update_status("Gray filter applied.")

    def invert_colors():
        if img:
            inverted_img = ImageOps.invert(img.convert("RGB"))
            show_image(inverted_img, panel_filtered)
            update_status("Inverted colors.")

    def apply_sharpen():
        if img:
            show_image(img.filter(ImageFilter.SHARPEN), panel_filtered)
            update_status("Sharpen filter applied.")

    def apply_gaussian_blur():
        if img:
            show_image(img.filter(ImageFilter.GaussianBlur(radius=2)), panel_filtered)
            update_status("Gaussian blur applied.")

    def apply_mean_filter():
        global img
        if not img:
            return

        try:
            size = int(mask_size_entry.get())
            if size % 2 == 0 or size < 1:
                raise ValueError("Size must be an odd positive integer.")
        except ValueError as e:
            update_status(f"Invalid mask size: {e}")
            return

        img_rgb = img.convert("RGB")
        img_array = np.array(img_rgb, dtype=np.float32)

        kernel = np.ones((size, size), dtype=np.float32) / (size * size)
        pad = size // 2

        padded = np.pad(img_array, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        result = np.zeros_like(img_array)

        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                for c in range(3):  # For R, G, B channels
                    region = padded[y:y+size, x:x+size, c]
                    result[y, x, c] = np.sum(region * kernel)

        result_img = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
        show_image(result_img, panel_filtered)
        update_status(f"Mean filter applied with size {size}x{size}.")

    def apply_median_filter():
        if img:
            show_image(img.filter(ImageFilter.MedianFilter(size=mask_size_entry.get())), panel_filtered)
            update_status("Median filter applied.")

    def add_gaussian_noise():
        global img
        if not img:
            update_status("No image loaded.")
            return

        img_gray = img.convert("L")
        img_array = np.array(img_gray, dtype=np.uint8)
        noise = np.random.normal(loc=0, scale=20, size=img_array.shape).astype(np.int16)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(noisy)
        show_image(result_img, panel_filtered)
        update_status("Gaussian noise added.")

    def add_salt_pepper_noise():
        global img
        if not img:
            update_status("No image loaded.")
            return

        img_gray = img.convert("L")
        img_array = np.array(img_gray)
        noise_pct = 0.1
        num_pixels = int(noise_pct * img_array.size)
        coords = np.random.choice(img_array.size, num_pixels, replace=False)
        salt_vs_pepper = np.random.choice([0, 255], size=num_pixels)
        flat_img = img_array.flatten()
        flat_img[coords] = salt_vs_pepper
        result_img = Image.fromarray(flat_img.reshape(img_array.shape).astype(np.uint8))
        show_image(result_img, panel_filtered)
        update_status("Salt & Pepper noise added.")

    def apply_midpoint_filter():
        if not img:
            update_status("No image loaded.")
            return

        img_gray = img.convert("L")
        img_array = np.array(img_gray)
        padded = np.pad(img_array, 1, mode='edge')
        filtered = np.zeros_like(img_array)

        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                region = padded[y:y+3, x:x+3]
                filtered[y, x] = (np.max(region) + np.min(region)) // 2

        result_img = Image.fromarray(filtered)
        show_image(result_img, panel_filtered)
        update_status("Midpoint filter applied.")

    def apply_geometric_mean_filter():
        if not img:
            update_status("No image loaded.")
            return

        img_gray = img.convert("L")
        img_array = np.array(img_gray, dtype=np.float32)
        padded = np.pad(img_array, 1, mode='edge')
        filtered = np.zeros_like(img_array)

        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                region = padded[y:y+3, x:x+3]
                product = np.prod(region)
                filtered[y, x] = np.power(product, 1.0 / 9)

        result_img = Image.fromarray(np.clip(filtered, 0, 255).astype(np.uint8))
        show_image(result_img, panel_filtered)
        update_status("Geometric mean filter applied.")

    def apply_min_filter():
        if not img:
            update_status("No image loaded.")
            return

        img_gray = img.convert("L")
        img_array = np.array(img_gray)
        padded = np.pad(img_array, 1, mode='edge')
        filtered = np.zeros_like(img_array)

        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                region = padded[y:y+3, x:x+3]
                filtered[y, x] = np.min(region)

        result_img = Image.fromarray(filtered)
        show_image(result_img, panel_filtered)
        update_status("Min filter applied.")

    def apply_max_filter():
        if not img:
            update_status("No image loaded.")
            return

        img_gray = img.convert("L")
        img_array = np.array(img_gray)
        padded = np.pad(img_array, 1, mode='edge')
        filtered = np.zeros_like(img_array)

        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                region = padded[y:y+3, x:x+3]
                filtered[y, x] = np.max(region)

        result_img = Image.fromarray(filtered)
        show_image(result_img, panel_filtered)
        update_status("Max filter applied.")

    def apply_arithmetic_mean_filter():
        if not img:
            update_status("No image loaded.")
            return

        img_gray = img.convert("L")
        img_array = np.array(img_gray, dtype=np.float32)
        padded = np.pad(img_array, 1, mode='edge')
        filtered = np.zeros_like(img_array)

        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                region = padded[y:y+3, x:x+3]
                filtered[y, x] = np.mean(region)

        result_img = Image.fromarray(np.clip(filtered, 0, 255).astype(np.uint8))
        show_image(result_img, panel_filtered)
        update_status("Arithmetic mean filter applied.")

    def apply_filter_to_region():
        global img, filtered_img, start_x, start_y, end_x, end_y
        if img and start_x is not None and end_x is not None:
            image = img.copy()
            scale_x = img.width / IMAGE_DISPLAY_SIZE[0]
            scale_y = img.height / IMAGE_DISPLAY_SIZE[1]
            x1 = int(min(start_x, end_x) * scale_x)
            y1 = int(min(start_y, end_y) * scale_y)
            x2 = int(max(start_x, end_x) * scale_x)
            y2 = int(max(start_y, end_y) * scale_y)
            region = image.crop((x1, y1, x2, y2))
            region_filtered = region.filter(ImageFilter.GaussianBlur(4))
            image.paste(region_filtered, (x1, y1))
            show_image(image, panel_filtered)
            update_status("Blurred selected region.")

    def histogram_equalization():
        global img
        if not img:
            update_status("No image loaded.")
            return

        gray_img = img.convert("L")
        img_array = np.array(gray_img)
        h, w = img_array.shape
        T = 256 / (w * h)

        frequency = [0] * 256
        for i in range(h):
            for j in range(w):
                frequency[img_array[i, j]] += 1

        cumulative_sum = [0] * 256
        cumulative_sum[0] = frequency[0]
        for k in range(1, 256):
            cumulative_sum[k] = cumulative_sum[k - 1] + frequency[k]

        for i in range(256):
            cumulative_sum[i] = min(255, int(cumulative_sum[i] * T))

        equalized = np.zeros_like(img_array)
        for i in range(h):
            for j in range(w):
                equalized[i, j] = cumulative_sum[img_array[i, j]]

        result_img = Image.fromarray(equalized)
        show_image(result_img, panel_filtered)
        update_status("histogram equalization applied.")

    def adjust_brightness():
        global img
        if not img:
            update_status("No image loaded.")
            return

        value = brightness_scale.get()
        img_rgb = img.convert("RGB")
        img_array = np.array(img_rgb, dtype=np.int16)

        img_array = np.clip(img_array + value, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(img_array)
        show_image(result_img, panel_filtered)
        update_status(f"Brightness adjusted by {value}")

    def adjust_contrast():
        global img
        if not img:
            update_status("No image loaded.")
            return

        value = contrast_scale.get()
        img_rgb = img.convert("RGB")
        img_array = np.array(img_rgb, dtype=np.float32) / 255.0

        factor = (259 * (value + 255)) / (255 * (259 - value)) if value != 0 else 1
        result = np.clip((img_array - 0.5) * factor + 0.5, 0, 1)
        result_img = Image.fromarray((result * 255).astype(np.uint8))
        show_image(result_img, panel_filtered)
        update_status(f"Contrast adjusted by {value}")

    def apply_selected_blur(blur_name):
        
        if blur_name == "Gaussian":
            apply_gaussian_blur()

        elif blur_name == "Mean":
            apply_mean_filter()

        elif blur_name == "Median":
            apply_median_filter()

        elif blur_name == "Region":
            apply_filter_to_region()

        else:
            update_status(f"Unknown blur type: {blur_name}")


    def update_status(message):
        status_bar.config(text=message)

    # ===================== GUI Layout =======================

    frame_images = Frame(window)
    frame_images.pack(pady=20)


    frame_original = Frame(frame_images, width=350, height=350, bd=2, relief="solid", pady=5)
    frame_original.pack(side="left", padx=20)
    frame_original.pack_propagate(False)

    frame_filtered = Frame(frame_images, width=350, height=350, bd=2, relief="solid", pady=5)
    frame_filtered.pack(side="right", padx=20)
    frame_filtered.pack_propagate(False)

    canvas = Canvas(frame_original, width=350, height=350, cursor="crosshair")
    canvas.pack()
    canvas.bind("<Button-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_drag)

    panel_filtered = Label(frame_filtered)
    panel_filtered.pack(expand=True)

    # Control Buttons
    frame_controls = Frame(window)
    frame_controls.pack(pady=20)

    btn_open = Button(frame_controls, text="Open Image", command=open_image)
    btn_open.grid(row=7, column=0, padx=10)

    btn_blue = Button(frame_controls, text="Blue Filter", command=blue_filter)
    btn_blue.grid(row=0, column=0, padx=10)

    btn_red = Button(frame_controls, text="Red Filter", command=red_filter)
    btn_red.grid(row=0, column=1, padx=10)

    btn_green = Button(frame_controls, text="Green Filter", command=green_filter)
    btn_green.grid(row=0, column=2, padx=10)

    btn_gray = Button(frame_controls, text="Gray Filter", command=gray_filter)
    btn_gray.grid(row=0, column=3, padx=10, pady=10)

    btn_invert = Button(frame_controls, text="Invert Colors", command=invert_colors)
    btn_invert.grid(row=1, column=0, padx=10, pady=10)

    btn_sharpen = Button(frame_controls, text="Sharpen", command=apply_sharpen)
    btn_sharpen.grid(row=1, column=1, padx=10, pady=10)

    # Dropdown for blur type
    blur_options = ["Gaussian", "Mean", "Median", "Region"]
    blur_type = StringVar(value=blur_options[0])

    Label(frame_controls, text="Blur Type:").grid(row=3, column=0, padx=10)
    OptionMenu(frame_controls, blur_type, *blur_options).grid(row=3, column=1, padx=10)

    # Entry for mask size
    Label(frame_controls, text="Mask Size:").grid(row=3, column=2, padx=10)
    mask_size_entry = Entry(frame_controls, width=5)
    mask_size_entry.grid(row=3, column=3, padx=10)

    # Apply blur button
    btn_apply_blur = Button(frame_controls, text="Apply Blur", command=lambda: apply_selected_blur(blur_type.get()))
    btn_apply_blur.grid(row=3, column=4, padx=10)

    btn_equalize = Button(frame_controls, text="Equalize Histogram", command=histogram_equalization)
    btn_equalize.grid(row=1, column=2, padx=10, pady=10)

    btn_gaussian_noise = Button(frame_controls, text="Gaussian Noise", command=add_gaussian_noise)
    btn_gaussian_noise.grid(row=6, column=0, padx=10, pady=5)

    btn_sp_noise = Button(frame_controls, text="Salt & Pepper Noise", command=add_salt_pepper_noise)
    btn_sp_noise.grid(row=6, column=1, padx=10, pady=5)

    btn_midpoint = Button(frame_controls, text="Midpoint Filter", command=apply_midpoint_filter)
    btn_midpoint.grid(row=6, column=2, padx=10, pady=5)

    btn_geo_mean = Button(frame_controls, text="Geometric Mean Filter", command=apply_geometric_mean_filter)
    btn_geo_mean.grid(row=6, column=3, padx=10, pady=5)

    btn_min_filter = Button(frame_controls, text="Min Filter", command=apply_min_filter)
    btn_min_filter.grid(row=5, column=0, padx=10, pady=5)

    btn_max_filter = Button(frame_controls, text="Max Filter", command=apply_max_filter)
    btn_max_filter.grid(row=5, column=1, padx=10, pady=5)

    btn_arithmetic_mean = Button(frame_controls, text="Arithmetic Mean Filter", command=apply_arithmetic_mean_filter)
    btn_arithmetic_mean.grid(row=5, column=2, padx=10, pady=5)
    # Brightness slider and button
    Label(frame_controls, text="Brightness:").grid(row=4, column=0, padx=10)
    brightness_scale = Scale(frame_controls, from_=-100, to=100, orient=HORIZONTAL)
    brightness_scale.set(0)
    brightness_scale.grid(row=4, column=2)

    btn_brightness = Button(frame_controls, text="Apply Brightness", command=adjust_brightness)
    btn_brightness.grid(row=4, column=3, padx=10)

    # Contrast slider and button
    Label(frame_controls, text="Contrast:").grid(row=4, column=0, padx=10)
    contrast_scale = Scale(frame_controls, from_=-100, to=100, orient=HORIZONTAL)
    contrast_scale.set(0)
    contrast_scale.grid(row=4, column=0)

    btn_contrast = Button(frame_controls, text="Apply Contrast", command=adjust_contrast)
    btn_contrast.grid(row=4, column=1, padx=10)

    btn_save = Button(frame_controls, text="Save Output", command=save_output)
    btn_save.grid(row=7, column=1, padx=10, pady=10)

    # Status bar
    status_bar = Label(window, text="Welcome!", bd=1, relief="sunken", anchor=W)
    status_bar.pack(side=BOTTOM, fill=X)

    window.mainloop()







