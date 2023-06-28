/** @type {import('next').NextConfig} */


const nextConfig = {
  experimental: {
    appDir: true,
  },
  webpack(config) {
    config.module.rules.push({
      test: /\.svg$/,
      use: ["@svgr/webpack"],
    });

    return config;
  },
  output: "standalone",
  // async rewrites(){
  //   return [
  //     {
  //       source:'/worker/:path*',
  //       destination:'http://localhost:21002/:path*',
  //     }
  //   ]
  // }
};

module.exports = nextConfig;
