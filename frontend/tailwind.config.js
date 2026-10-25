/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cream: {
          DEFAULT: "#F3EEE4",
          light: "#FAF7F2",
          dark: "#E8E2D5",
        },
        espresso: {
          DEFAULT: "#2B2420",
          dark: "#1F1A17",
          muted: "#3D342F",
        },
        olive: {
          DEFAULT: "#8A9A5B",
          light: "#9EAE6F",
          dark: "#75844D",
        },
      },
      fontFamily: {
        display: ["Fraunces", "Georgia", "serif"],
        body: ["Work Sans", "Inter", "system-ui", "sans-serif"],
        sans: ["Work Sans", "Inter", "system-ui", "sans-serif"],
        pixel: ["BubbledotICG-FinePos", "Geist Pixel Circle", "monospace"],
      },
      lineHeight: {
        headline: "0.95",
      },
      letterSpacing: {
        "tight-headline": "-0.045em",
        "widest-label": "0.18em",
      },
      fontSize: {
        headline: [
          "clamp(2.75rem, 5.5vw, 4.25rem)",
          { lineHeight: "0.95", letterSpacing: "-0.045em" },
        ],
        label: [
          "0.75rem",
          { lineHeight: "1rem", letterSpacing: "0.18em" },
        ],
      },
    },
  },
  plugins: [],
};
