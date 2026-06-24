/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Ink-dark canvas: a near-black with a faint blue cast.
        ink: {
          950: "#06070b",
          900: "#0a0c12",
          800: "#0f121b",
          700: "#161a26",
          600: "#1e2433",
        },
        line: "#232a3b", // hairline borders
        muted: "#7c869b", // secondary text
        fg: "#e7eaf2", // primary text
        accent: "#67e8f9", // single restrained glow color (cyan)
        "accent-dim": "#2a4d57",
      },
      fontFamily: {
        sans: ['"Inter"', "system-ui", "-apple-system", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 24px -4px rgba(103,232,249,0.35)",
      },
    },
  },
  plugins: [],
};
