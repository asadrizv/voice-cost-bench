/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  reactStrictMode: true,
  // A build sharing `.next` with a running `next dev` corrupts the dev server's chunks.
  distDir: process.env.NEXT_DIST_DIR ?? ".next",
};
export default nextConfig;
