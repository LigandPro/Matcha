import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  reactStrictMode: true,
  sassOptions: {
    includePaths: ['node_modules'],
  },
};

export default nextConfig;
