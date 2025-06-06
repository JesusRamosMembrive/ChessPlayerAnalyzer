/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}", // This line ensures Tailwind scans all JS/JSX/TS/TSX files in src
    "./public/index.html", // Also scan index.html if Tailwind classes are used there
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
