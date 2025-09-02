/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    // Enable if needed
  },
  // Ensure webpack can resolve the @ alias properly
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      '@': require('path').resolve(__dirname),
    }
    return config
  }
}

module.exports = nextConfig