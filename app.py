import tkinter as tk
from image_sketch import ImageSketchApp


def main():
    root = tk.Tk()
    app = ImageSketchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
