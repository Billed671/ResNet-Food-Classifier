import { fireEvent, render, screen } from "@testing-library/react";
import App from "./App.jsx";

// jsdom does not implement object URLs.
URL.createObjectURL = () => "blob:preview";
URL.revokeObjectURL = () => {};

afterEach(() => {
  vi.unstubAllGlobals();
});

function chooseImage() {
  const image = new File(["pixels"], "dish.jpg", { type: "image/jpeg" });
  fireEvent.change(screen.getByLabelText("Food image"), { target: { files: [image] } });
}

test("shows the predicted class and confidence", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ label: "spaghetti_carbonara", confidence: 0.873 }),
    }),
  );
  render(<App />);

  chooseImage();
  fireEvent.click(screen.getByRole("button", { name: "Classify" }));

  expect(await screen.findByText("spaghetti carbonara")).toBeTruthy();
  expect(screen.getByText("87.3% confidence")).toBeTruthy();
  expect(screen.getByAltText("Selected dish")).toBeTruthy();
});

test("shows the error returned by the server", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: "File is not a valid image" }),
    }),
  );
  render(<App />);

  chooseImage();
  fireEvent.click(screen.getByRole("button", { name: "Classify" }));

  expect((await screen.findByRole("alert")).textContent).toBe("File is not a valid image");
});

test("rejects files that are not images", () => {
  render(<App />);

  fireEvent.change(screen.getByLabelText("Food image"), {
    target: { files: [new File(["text"], "notes.txt", { type: "text/plain" })] },
  });

  expect(screen.getByRole("alert").textContent).toBe("Please choose an image file.");
  expect(screen.getByRole("button", { name: "Classify" }).disabled).toBe(true);
});
