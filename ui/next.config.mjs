/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:2024",
    NEXT_PUBLIC_ASSISTANT_ID: process.env.NEXT_PUBLIC_ASSISTANT_ID || "trip_planner",
  },
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
};

export default nextConfig;
